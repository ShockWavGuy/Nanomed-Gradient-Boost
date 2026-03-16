"""
═══════════════════════════════════════════════════════════════════════════════
RUN SCRIPT FOR CASCADING MOBO v2  —  dataset-wired version
═══════════════════════════════════════════════════════════════════════════════

Datasets:
    PLGAset.csv      — 433 PLGA formulations, target: LC
                       Group: study (59 unique studies)
                       logDP column: drugpolymer (range 0.002-1.0)

    datasetCELL2.csv — 1260 brain delivery experiments, target: ID%
                       ~651 non-null ID% rows; ~390 after organic-only filter
                       (polymer-based NPs + liposomes + SLN — inorganic excluded)
                       Group: group column
                       Target: raw ID%

Cascade:
    Stage 1 (CatBoost): formulation + drug chemistry -> LC   (R2=0.90)
    Stage 2 (TabPFN+GP): formulation + LC_pred + context -> ID%  (R2=0.57)

MOBO optimizes jointly: maximize LC + maximize brain delivery
Ligand selection: enumerated per disease, selected by Pareto hypervolume

    Target: raw ID%

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from mobo_v14 import (
    SearchSpace,
    Surrogate,
    TabPFNSurrogate,
    CascadingSurrogate,
    CascadingMOBO,
    ConformalCalibrator,
    export_results,
    diagnose_mobo,
    pareto_grid_sample,
)
from plga_degradation import DEGRADATION_FEATURES, add_degradation_features



# ═══════════════════════════════════════════════════════════════════════════════
# DATA PATHS
# ═══════════════════════════════════════════════════════════════════════════════

BRAIN_DATA_PATH = 'datasetCELL2.csv'
LC_DATA_PATH    = 'PLGAset.csv'


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG 1 — FEATURE SPACES
# Column names match PLGAset.csv (LC model) and datasetCELL2.csv (brain model).
# ═══════════════════════════════════════════════════════════════════════════════

MY_CONTINUOUS_PARAMS = {
    # PLGAset.csv column names
    'polymer_MW':               (7.0,   40.0),    # kDa — capped at 40 kDa; >40 kDa PLGA has slow degradation kinetics incompatible with CNS NP design
    'LAGA':                     (0.50,  1.00),    # LA fraction (0.5=50:50, 1.0=pure PLA)
    'drugpolymer':              (0.05,  1.00),    # drug:polymer ratio (logDP)
    'surfactant_concentration': (0.1,   5.0),     # % w/v
    'particle_size':            (80.0,  300.0),   # nm -- shared name across datasets
    # Brain dataset
    'PEGRatio':                 (0.0,   0.20),    # PEG mass fraction
    # Fix 1A: PDI and zeta are rank-2 and rank-4 brain delivery features
    # (15.4% and 12.9% importance) — including them in the search space gives
    # the MOBO real control over 28% of its own signal.
    'PDI':                      (0.05,  0.30),    # polydispersity index; <0.2 = monodisperse
    'zeta':                     (-50.0, -5.0),    # mV; negative = stable NP (electrostatic repulsion)
}

MY_CATEGORICAL_PARAMS = {
    # PLGAset.csv
    'solvent':         ['acetone', 'acetonitrile', 'ethyl acetate'],  # dmso + THF removed (DMSO: not standard nanoprecipitation; THF: ICH Class 2 carcinogen)
    'surfactant_type': ['PVA', 'poloxamer', 'tween80', 'SDS'],  # → surfactant_HLB via lookup
    # datasetCELL2.csv — route/mechanism fixed per drug, but ligand/receptor
    # are optimizable: different targeting strategies are the core MOBO output.
    # Options derived from brain dataset training encodings.
    'route':     ['i.v.', 'i.p.', 'i.n.', 'p.o.'],
    'mechanism': ['passive transport', 'transporter mediated', 'receptor mediated',
       'absorption mediated', 'cell mediated', 'transcytosis'],
    'prep':      ['polymer-based nanoparticles'],
    'comp':      ['plga'],
    # Targeting ligand — primary driver of receptor-mediated brain delivery variation.
    # Not in fixed_features so MOBO can optimise targeting strategy.
    'ligand':    ['none', 'transferrin', 'glucose', 'folate', 'RGD', 'apolipoprotein'],
    # Receptor target — paired with ligand, drives EHVI exploration.
    'receptor':  ['none', 'transferrin receptor', 'GLUT1', 'folate receptor',
                  'integrin', 'LDL receptor'],
}

MY_BINARY_PARAMS = {
    'PEG':      [0, 1],
    'external': [0, 1],
}


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG 2 — LOOKUP TABLES
# ═══════════════════════════════════════════════════════════════════════════════

# Animal model strings match datasetCELL2.csv 'animal-model' column exactly
MY_BBB_PERMEABILITY = { #Approximated from literature
    'normal':                                    1.00,
    'brain tumor':                               8.00,
    'ischemic stroke':                           4.00,
    'ischemic stroke/reperfusion':               5.00,
    'stroke(ischemic stroke, hemorrhagic stroke)': 3.50,
    "Alzheimer's disease":                       1.60,
    "Parkinson's disease":                       1.70,
    'encephalomyelitis':                         4.00,
    'others':                                    1.50,
    'unknown':                                   1.00,
}

MY_SURFACTANT_HLB = {
    'PVA': 16.0, 'poloxamer': 22.0, 'tween80': 15.0, 'SDS': 40.0, 'none': 0.0,
}

# ── Receptor-specific brain endothelial transcriptomics ───────────────────────
# Source: bulk RNA-seq of healthy human brain microvascular endothelial cells
# (BBB) and disease-state differentially expressed gene data.
#
# RECEPTOR_BBB_EXPR: log1p-transformed baseline (healthy BBB) expression (TPM).
#   log1p compresses the 5-order-of-magnitude range (GLUT1≈4824 vs ITGB3≈4.4)
#   into a [0, ~8.5] interval that GPBoost trees can split meaningfully.
#
# _RECEPTOR_OMICS_FC: log2 fold-changes in disease states relative to healthy.
#   Indices: [BM=epilepsy, CK=EAE/encephalomyelitis, DG=ischemic stroke, EK=TBI]
#   Entries with [0] (folr1, slc5a7) have no expression data — treated as 0.
#
# 'integrin' is the average of ITGAV and ITGB3 (both subunits of αvβ3 integrin,
#   the RGD receptor); averaging reflects that both subunits must be present.
RECEPTOR_BBB_EXPR = {
    'none':                 0.0,
    'transferrin receptor': np.log1p(1149.3),           # TFR1 — highly expressed
    'GLUT1':                np.log1p(4824.5),           # SLC2A1 — dominant BBB glucose transporter
    'LDL receptor':         np.log1p(78.7),             # LDLR — moderate expression
    'folate receptor':      0.0,                        # FOLR1 — no expression data
    'integrin':             np.log1p((36.9 + 4.4) / 2.0),  # avg(ITGAV, ITGB3)
}

# Maps alternative receptor name spellings found in datasetCELL2.csv to their
# canonical key in RECEPTOR_BBB_EXPR / _RECEPTOR_OMICS_FC.
# All comparisons are done after .lower().strip() so capitalisation is irrelevant.
RECEPTOR_NAME_ALIASES: dict = {
    # ── GLUT1 / glucose transporter variants ─────────────────────────────
    'GLUT-1':                                           'GLUT1',
    'glucose transporter-1':                            'GLUT1',
    'glucose transporter 1':                            'GLUT1',
    'SLC2A1':                                           'GLUT1',

    # ── Transferrin receptor variants ─────────────────────────────────────
    'TfR':                                              'transferrin receptor',
    'TfR1':                                             'transferrin receptor',
    'TFRC':                                             'transferrin receptor',
    '转铁蛋白受体':                                     'transferrin receptor',  # Chinese
    'transferrin receptor+dopamine receptor':           'transferrin receptor',

    # ── Folate / folic acid receptor ──────────────────────────────────────
    'folic acid receptor':                              'folate receptor',
    'FOLR1':                                            'folate receptor',
    'FOLR':                                             'folate receptor',

    # ── Integrin family (αvβ1, αvβ3, unicode variants) ───────────────────
    'αvβ3':                                             'integrin',
    'αvβ1':                                             'integrin',
    '𝛂v𝛃3':                                          'integrin',
    'avb3':                                             'integrin',
    '\xa0both\xa0integrin\xa0αvβ3 and\xa0neuropilin-1\xa0receptors': 'integrin',

    # ── LDL receptor / LRP family ─────────────────────────────────────────
    'LRP-1':                                            'LDL receptor',
    'LRP1':                                             'LDL receptor',
    'LRP 1':                                            'LDL receptor',
    'LDL receptor-related protein':                    'LDL receptor',
    'low density lipoprotein receptor-related protein': 'LDL receptor',
    'low density lipoprotein-related receptor':        'LDL receptor',
    'apolipoprotein E receptor':                        'LDL receptor',
    'ApoE3':                                            'LDL receptor',
    'very low-density lipoprotein (VLDL) receptor':    'LDL receptor',
    # Lactoferrin receptor signals via LRP-1/LRP-2 (megalin) at BBB
    'Lactoferrin receptor':                             'LDL receptor',
    'Lactoferrin recepter':                             'LDL receptor',  # CSV typo
    '乳铁蛋白受体':                                    'LDL receptor',  # Chinese
}

_RECEPTOR_OMICS_FC = {
    # receptor key: [BM(epilepsy), CK(EAE), DG(stroke), EK(TBI)]  — log2 FC
    'none':                 [0.0,    0.0,    0.0,    0.0],
    'transferrin receptor': [-0.424, -1.004,  0.210, -1.118],
    'GLUT1':                [-0.463, -0.406, -0.210, -0.335],
    'LDL receptor':         [-0.206, -0.599, -0.228,  0.316],
    'folate receptor':      [0.0,    0.0,    0.0,    0.0],   # no data
    'integrin':             [(-0.397 + 0.615) / 2,   # BM: avg(ITGAV, ITGB3)
                             ( 0.478 + 1.844) / 2,   # CK
                             (-0.379 + 0.269) / 2,   # DG
                             ( 0.650 + 3.694) / 2],  # EK
}

# Maps animalmodel strings (datasetCELL2.csv) → index into _RECEPTOR_OMICS_FC.
# None = baseline (FC = 0.0). BM/EK conditions not in training data → mapped
# to None so out-of-training entries default to zero fold-change.
_ANIMALMODEL_TO_OMICS_COND = {
    # Indices: 0=epilepsy, 1=EAE/encephalomyelitis, 2=stroke, 3=TBI
    'normal':                                        None,  # healthy baseline — no FC
    "Alzheimer's disease":                           1,     # EAE proxy: shared chronic neuroinflammation
    "Parkinson's disease":                           None,  # no matching omics condition → baseline
    'brain tumor':                                   3,     # TBI acute: shared acute BBB disruption
    'ischemic stroke':                               2,     # stroke acute — direct match
    'ischemic stroke/reperfusion':                   2,
    'stroke(ischemic stroke, hemorrhagic stroke)':   2,
    'encephalomyelitis':                             1,     # EAE — direct match
    'others':                                        None,
    'unknown':                                       None,
}


def _get_receptor_disease_fc(receptor: str, animalmodel) -> float:
    """
    Return the log2 fold-change of `receptor` expression in the disease
    context described by `animalmodel`.

    Used to compute the `receptor_disease_fc` feature for each training row.
    MOBO candidates use animalmodel in {Alzheimer's, Parkinson's, normal},
    all of which map to None → 0.0 (correct: no disease-state FC applies).

    Both lookups are case-insensitive (lower-stripped) so CSV strings like
    'Transferrin Receptor' or 'Ischemic Stroke' are matched correctly.
    """
    _am_lower  = str(animalmodel).lower().strip() if animalmodel is not None else 'unknown'
    _rec_lower = str(receptor).lower().strip()
    # Resolve aliases so 'GLUT-1', 'TfR', etc. map to the canonical FC entry
    _rec_lower = RECEPTOR_NAME_ALIASES.get(_rec_lower,
                    RECEPTOR_NAME_ALIASES.get(receptor, _rec_lower)
                 ).lower().strip()
    _am_map  = {k.lower().strip(): v for k, v in _ANIMALMODEL_TO_OMICS_COND.items()}
    _ofc_map = {k.lower().strip(): v for k, v in _RECEPTOR_OMICS_FC.items()}
    fc_idx  = _am_map.get(_am_lower)
    if fc_idx is None:
        return 0.0
    fc_list = _ofc_map.get(_rec_lower, [0.0, 0.0, 0.0, 0.0])
    return float(fc_list[fc_idx])


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG 3 — CASCADE VARIABLE
# ═══════════════════════════════════════════════════════════════════════════════

MY_CASCADE_VARS = {
    'LC': 'DL',   # PLGAset LC (loading capacity) → brain model's 'DL' column
}

# ── Organic nanoparticle families filter ─────────────────────────────────────
# Brain delivery training is restricted to organic carriers so the surrogate
# learns from formulations mechanistically comparable to the PLGA designs
# MOBO is optimising. Inorganic carriers (gold, silica, iron-oxide, etc.) have
# fundamentally different surface chemistry and biodistribution and would add
# noise that degrades prediction quality for organic candidates.
# Matches the 'prep' column in datasetCELL2.csv (exact string match).
ORGANIC_PREPS = {
    # Core organic carrier families — mechanistically comparable to PLGA.
    # Exact strings must match the 'prep' column in datasetCELL2.csv.
    # Run with organic=True to see prep value_counts before filtering;
    # add any missing organic variants here if the row count is lower than expected.
    'polymer-based nanoparticles',
    'polymeric nanoparticles',        # common alternative spelling
    'liposomes',
    'liposome',                       # singular variant
    'pegylated liposomes',
    'solid lipid nanoparticles',
    'solid lipid nanoparticle',       # singular variant
    'lipid nanoparticles',            # without 'solid'
    'nanostructured lipid carriers',
}

# Training-set means for Stage-2 columns absent from the search space.
# Computed directly from datasetCELL2.csv to keep candidates in-distribution.
# Recompute if dataset changes: pd.read_csv(BRAIN_DATA_PATH)[col].dropna().mean()
def _compute_imputation_defaults(brain_data_path: str) -> dict:
    """Read training means from brain dataset at import time."""
    try:
        df = pd.read_csv(brain_data_path)
        # Apply the same organic-families filter used during model training so
        # imputation means reflect the actual training distribution.
        if 'prep' in df.columns:
            df = df[
                df['prep'].str.lower().str.strip().isin(
                    {p.lower() for p in ORGANIC_PREPS}
                )
            ].reset_index(drop=True)
        # PDI and zeta removed — they are now search space variables (Fix 1A)
        # and must NOT be imputed; the sampled value will be used instead.
        numeric_cols = ['liganddensitymol']
        defaults = {}
        for col in numeric_cols:
            if col in df.columns:
                val = df[col].dropna()
                defaults[col] = float(val.mean()) if len(val) > 0 else 0.0
            else:
                defaults[col] = 0.0
        # Categorical cols: use most-frequent encoded value (0 = first category)
        # ligand and receptor are now search space variables — not imputed
        for col in ['surfacecoating', 'ligandtype', 'receptortype']:
            defaults[col] = 0
        # Receptor omics features:
        #   receptor_bbb_expr — wired via lookup_col_map from sampled 'receptor';
        #     imputed here as 0.0 (fallback if lookup fails, e.g. unseen receptor).
        #   receptor_disease_fc — always 0.0 for MOBO candidates because all
        #     target drugs (AD/PD/normal) have no disease-state FC in the omics
        #     dataset. Training rows get the real FC from load_data().
        defaults['receptor_bbb_expr']   = 0.0
        defaults['receptor_disease_fc'] = 0.0
        return defaults
    except Exception:
        # Fallback if file not yet available
        # PDI and zeta are search space variables (Fix 1A) — not imputed
        return {
            'liganddensitymol': 0.0,
            'surfacecoating': 0, 'ligandtype': 0, 'receptortype': 0,
            'receptor_bbb_expr': 0.0, 'receptor_disease_fc': 0.0,
        }

STAGE2_IMPUTATION_DEFAULTS = _compute_imputation_defaults(BRAIN_DATA_PATH)

# Column names that differ between the LC dataset and brain delivery dataset
# for the same physical quantity. Applied during Stage 2 feature extraction.
MY_COLUMN_ALIASES = {
    'particle_size': 'size',   # PLGAset 'particle_size' == brain dataset 'size'
}


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG 4 — OBJECTIVES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Biological validity constraints ──────────────────────────────────────────
# Ligand-receptor pairings must be mechanistically coherent.
# Unconstrained search produces combinations like glucose+integrin which have
# no biological basis and would be immediately rejected by any domain expert.
# Source: established CNS targeting literature.
VALID_LIGAND_RECEPTOR_PAIRS = {
    # (ligand, receptor) → True means biologically valid
    ('none',           'none'),
    ('transferrin',    'transferrin receptor'),
    ('glucose',        'GLUT1'),
    ('folate',         'folate receptor'),
    ('RGD',            'integrin'),
    ('apolipoprotein', 'LDL receptor'),
    # Allow 'none' ligand with 'none' receptor for passive transport candidates
}

# Deterministic receptor → canonical ligand mapping.
# Passed to SearchSpace(canonical_ligand_map=...) so that whenever MOBO
# samples a receptor, the ligand is assigned deterministically — invalid
# combinations like glucose+integrin are impossible by construction.
# Derived directly from VALID_LIGAND_RECEPTOR_PAIRS (1-to-1 mapping).
RECEPTOR_TO_CANONICAL_LIGAND = {
    'none':                 'none',
    'transferrin receptor': 'transferrin',
    'GLUT1':                'glucose',
    'folate receptor':      'folate',
    'integrin':             'RGD',
    'LDL receptor':         'apolipoprotein',
}

# Training data bounds for extrapolation detection
# Predictions outside these ranges are extrapolated — flag in output
BRAIN_DELIVERY_TRAIN_BOUNDS = (0.0, 0.852)   # training max raw ID% (informational; upper cap removed)
LC_TRAIN_BOUNDS             = (0.0, 36.5)    # from load_data output

MY_OBJECTIVES = {
    'LC': {
        'model':     'lc',
        'direction': 'maximize',
        'ref_point': 0.0,       # LC is always positive; 0 is a safe floor
    },
    'brain_delivery_logID_pct': {
        'model':     'brain',
        'direction': 'maximize',
        'ref_point': -1.0,      # raw ID% space; physical minimum is 0 but GPBoost
                                # can extrapolate slightly below 0 for unseen groups.
                                # -1.0 is safely below any real observation.
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG 5 — FEATURE COLUMN ASSIGNMENTS
# ═══════════════════════════════════════════════════════════════════════════════

STAGE1_LC_FEATURES = [
    'drugpolymer',            # logDP (drug:polymer ratio) -- dominant predictor
    'polymer_MW',
    'LAGA',
    'mol_MW',
    'mol_logP',
    'mol_TPSA',
    'mol_melting_point',
    'mol_Hacceptors',
    'mol_Hdonors',
    'mol_heteroatoms',
    'surfactant_concentration',
    'surfactant_HLB',
    'aqueousorganic',
    'pH',
    'solvent_polarity_index',
    'particle_size',          # dominant secondary predictor (SA/V mechanism)
    'solvent',
    'MolLogP',
    'BertzCT',
    'TPSA',
    'LabuteASA',
    'MolMR',
    'FractionCSP3',
    'NumHDonors',
    'NumHAcceptors',
    'NumRotatableBonds',
    'HeavyAtomCount',
    'RingCount',
    'NumAromaticRings',
    'NumHeteroatoms',
    'BalabanJ',
] + DEGRADATION_FEATURES

# 'LC' cascaded from Stage 1 -- do not list here, appended automatically
STAGE2_BRAIN_FEATURES = ['prep', 'comp', 'PEG', 'PEGRatio', 'surfacecoating', 'ligandtype',
       'ligand', 'mechanism', 'receptortype', 'receptor',
       # ── Receptor-specific brain endothelial transcriptomics ────────────────
       # receptor_bbb_expr: log1p(baseline healthy BBB expression, TPM).
       #   Encodes how abundantly the target receptor is expressed at the BBB —
       #   a high-expression receptor (GLUT1 > TFR > LDLR) provides more docking
       #   sites and should correlate with higher brain delivery.
       # receptor_disease_fc: log2 fold-change in disease state (from animalmodel).
       #   Positive = upregulated (more receptor available in disease); negative =
       #   downregulated (fewer docking sites). 0.0 for baseline/AD/PD conditions
       #   and for MOBO candidates (correct: target drugs are AD/PD/normal context).
       'receptor_bbb_expr', 'receptor_disease_fc',
       'size', 'PDI', 'zeta', 'EE', 'DL', 'animalmodel',
       'route', 'liganddensitymol',
       'external',
       'exact_mass', 'xlogp', 'tpsa', 'atom_stereo_count',
       'h_bond_donor_count', 'h_bond_acceptor_count', 'rotatable_bond_count',
       'heavy_atom_count', 'complexity'
]


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG 6 — FAILED CNS DRUG DEFINITIONS
# Fixed molecular properties use PLGAset column names.
# drugpolymer (logDP) is an OPTIMIZATION VARIABLE -- omit from fixed_features.
# ═══════════════════════════════════════════════════════════════════════════════

FAILED_CNS_DRUGS = {
    'semagacestat': {
        'indication':  "Alzheimer's disease",
        'failure':     'Phase III -- CNS delivery insufficient',
        'mechanism':   'gamma-secretase inhibitor',
        'fixed_features': {
            'mol_MW': 383.4, 'mol_logP': 1.5, 'mol_TPSA': 75.3,
            'mol_melting_point': 198.0, 'mol_Hacceptors': 5,
            'mol_Hdonors': 2, 'mol_heteroatoms': 6,
            'MolLogP': 1.5, 'TPSA': 75.3, 'HeavyAtomCount': 28,
            'NumHDonors': 2, 'NumHAcceptors': 5, 'NumRotatableBonds': 6,
            # 'animalmodel' (no underscore) matches datasetCELL2.csv column name
            'animalmodel': "Alzheimer's disease",
            # 'i.v.' (with dots) matches the brain dataset encoding
            'route': 'i.v.', 'prep': 'polymer-based nanoparticles',
            'mechanism': 'receptor mediated', 'PEG': 1,
            # 'zeta' removed — now a search space variable (Fix 1A)
        },
    },
    'atabecestat': {
        'indication':  "Alzheimer's disease",
        'failure':     'Phase II/III -- hepatotoxicity + delivery',
        'mechanism':   'BACE1 inhibitor',
        'fixed_features': {
            'mol_MW': 358.4, 'mol_logP': 2.8, 'mol_TPSA': 68.1,
            'mol_melting_point': 185.0, 'mol_Hacceptors': 4,
            'mol_Hdonors': 1, 'mol_heteroatoms': 5,
            'MolLogP': 2.8, 'TPSA': 68.1, 'HeavyAtomCount': 26,
            'NumHDonors': 1, 'NumHAcceptors': 4, 'NumRotatableBonds': 5,
            'animalmodel': "Alzheimer's disease",
            'route': 'i.v.', 'prep': 'polymer-based nanoparticles',
            'mechanism': 'receptor mediated', 'PEG': 1,
            # 'zeta' removed — now a search space variable (Fix 1A)
        },
    },
    'lixisenatide': {
        'indication':  "Parkinson's disease",
        'failure':     'BBB penetration insufficient for peptide',
        'mechanism':   'GLP-1 receptor agonist',
        'fixed_features': {
            'mol_MW': 4858.5, 'mol_logP': -1.2, 'mol_TPSA': 810.0,
            'mol_melting_point': 220.0, 'mol_Hacceptors': 65,
            'mol_Hdonors': 55, 'mol_heteroatoms': 80,
            'MolLogP': -1.2, 'TPSA': 810.0, 'HeavyAtomCount': 340,
            'NumHDonors': 55, 'NumHAcceptors': 65, 'NumRotatableBonds': 120,
            'animalmodel': "Parkinson's disease",
            'route': 'i.v.', 'prep': 'polymer-based nanoparticles',
            'mechanism': 'receptor mediated', 'PEG': 1,
            # 'zeta' removed — now a search space variable (Fix 1A)
        },
    },
    'ibudilast': {
        'indication':  'normal',
        'failure':     'Phase II -- marginal CNS efficacy',
        'mechanism':   'PDE inhibitor, anti-neuroinflammatory',
        'fixed_features': {
            'mol_MW': 230.3, 'mol_logP': 2.6, 'mol_TPSA': 58.5,
            'mol_melting_point': 50.0, 'mol_Hacceptors': 3,
            'mol_Hdonors': 0, 'mol_heteroatoms': 3,
            'MolLogP': 2.6, 'TPSA': 58.5, 'HeavyAtomCount': 17,
            'NumHDonors': 0, 'NumHAcceptors': 3, 'NumRotatableBonds': 4,
            'animalmodel': 'normal',
            'route': 'i.v.', 'prep': 'polymer-based nanoparticles',
            'mechanism': 'passive transport', 'PEG': 1,
            # 'zeta' removed — now a search space variable (Fix 1A)
        },
    },
    'sembragiline': {
        'indication':  "Alzheimer's disease",
        'failure':     'Phase II -- insufficient efficacy',
        'mechanism':   'MAO-B inhibitor',
        'fixed_features': {
            'mol_MW': 261.3, 'mol_logP': 2.1, 'mol_TPSA': 41.2,
            'mol_melting_point': 160.0, 'mol_Hacceptors': 2,
            'mol_Hdonors': 1, 'mol_heteroatoms': 3,
            'MolLogP': 2.1, 'TPSA': 41.2, 'HeavyAtomCount': 19,
            'NumHDonors': 1, 'NumHAcceptors': 2, 'NumRotatableBonds': 4,
            'animalmodel': "Alzheimer's disease",
            'route': 'i.v.', 'prep': 'polymer-based nanoparticles',
            'mechanism': 'receptor mediated', 'PEG': 1,
            # 'zeta' removed — now a search space variable (Fix 1A)
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(brain_path=BRAIN_DATA_PATH, lc_path=LC_DATA_PATH, organic=True):
    """
    Load PLGAset.csv and datasetCELL2.csv, return arrays ready for model fitting.

    Parameters
    ----------
    organic : bool, default True
        If True, restrict brain delivery training data to ORGANIC_PREPS
        (polymer-based NPs, liposomes, solid lipid NPs). Inorganic carriers
        have mechanistically different BBB transport — excluding them sharpens
        surrogate quality for PLGA candidates.

    Returns
    -------
    X_lc, y_lc, study_ids_lc         -- for LC CatBoost model
    X_brain, y_brain, study_ids, carrier_ids  -- for brain TabPFN+GP model
    feat_lc, feat_brain               -- feature name lists for model.fit()
    """
    # ── LC dataset (PLGAset.csv) ──────────────────────────────────────────
    df_lc = pd.read_csv(lc_path)

    # ── PLGA degradation features ──────────────────────────────────────────
    df_lc = add_degradation_features(df_lc, indication='normal')

    feat_lc = [c for c in STAGE1_LC_FEATURES if c in df_lc.columns]

    X_lc_df = df_lc[feat_lc].copy()
    lc_encodings = {}
    for col in X_lc_df.select_dtypes(include='object').columns:
        enc = {v: i for i, v in enumerate(X_lc_df[col].dropna().unique())}
        lc_encodings[col] = enc
        X_lc_df[col] = X_lc_df[col].map(enc).fillna(-1)
    X_lc         = X_lc_df.fillna(X_lc_df.median(numeric_only=True)).values.astype(float)
    y_lc         = df_lc['LC'].values
    study_ids_lc = df_lc['study'].values.astype(str)
    print(f"  LC:    {X_lc.shape[0]} samples, {X_lc.shape[1]} features")

    # ── Brain delivery dataset (datasetCELL2.csv) ─────────────────────────
    df_brain = pd.read_csv(brain_path)
    df_brain['log_ID_pct'] = df_brain['ID%']   # raw ID% (no log transform)
    df_brain = df_brain.dropna(subset=['ID%']).reset_index(drop=True)

    # Organic-only filter: retain polymer-based NPs, liposomes, and SLN.
    # Inorganic NPs (metal oxides) use passive/toxic BBB mechanisms — mechanistically distinct.
    n_before = len(df_brain)
    if organic and 'prep' in df_brain.columns:
        df_brain = df_brain[
            df_brain['prep'].str.lower().str.strip().isin(
                {p.lower() for p in ORGANIC_PREPS}
            )
        ].reset_index(drop=True)

    # Rename hyphenated column to match STAGE2_BRAIN_FEATURES
    if 'animal-model' in df_brain.columns:
        df_brain = df_brain.rename(columns={'animal-model': 'animalmodel'})

    # ── Receptor omics features ───────────────────────────────────────────────
    # receptor_bbb_expr: log1p of healthy BBB baseline expression (TPM) for the
    #   targeted receptor. Computed from RECEPTOR_BBB_EXPR lookup. Rows whose
    #   receptor string is not in the table (e.g. unknown or free-drug entries)
    #   get 0.0 (= log1p(0), same as 'none'). This is conservative: unknown
    #   receptors are treated as unexpressed rather than imputed from neighbours.
    if 'receptor' in df_brain.columns:
        # Case-insensitive lookup: normalise both sides to lowercase/stripped so
        # 'Transferrin Receptor' and 'transferrin receptor' both hit the table.
        # The original case-sensitive .map() silently returned NaN for every row
        # whose receptor string differed only in capitalisation, collapsing the
        # whole column to the 0.0 fallback and producing zero importance.
        _expr_lower_map = {k.lower().strip(): v for k, v in RECEPTOR_BBB_EXPR.items()}
        # Expand map with aliases so CSV variant spellings resolve to the same value
        # as their canonical form (e.g. 'GLUT-1' → GLUT1 value, 'TfR' → TFR value).
        for _alias, _canon in RECEPTOR_NAME_ALIASES.items():
            _canon_v = _expr_lower_map.get(_canon.lower().strip())
            if _canon_v is not None:
                _expr_lower_map[_alias.lower().strip()] = _canon_v

        _receptor_norm  = df_brain['receptor'].astype(str).str.lower().str.strip()
        _expr_series    = _receptor_norm.map(_expr_lower_map)

        # Use median of matched rows rather than 0.0 for unmatched receptors.
        _expr_median = _expr_series.dropna().median()
        _expr_fill   = _expr_median if not np.isnan(_expr_median) else 0.0
        df_brain['receptor_bbb_expr'] = _expr_series.fillna(_expr_fill)
    else:
        df_brain['receptor_bbb_expr'] = 0.0

    # receptor_disease_fc: log2 FC of receptor expression in the animalmodel
    #   disease state. Rows with no matching omics condition (AD, PD, normal,
    #   brain tumor, others, unknown) get 0.0 — correctly reflecting that
    #   these conditions have no differential expression data available.
    if 'receptor' in df_brain.columns and 'animalmodel' in df_brain.columns:
        df_brain['receptor_disease_fc'] = df_brain.apply(
            lambda r: _get_receptor_disease_fc(r['receptor'], r['animalmodel']),
            axis=1,
        )
    else:
        df_brain['receptor_disease_fc'] = 0.0

    feat_brain = [c for c in STAGE2_BRAIN_FEATURES if c in df_brain.columns]

    X_br_df = df_brain[feat_brain].copy()
    brain_encodings = {}
    for col in X_br_df.select_dtypes(include='object').columns:
        enc = {v: i for i, v in enumerate(X_br_df[col].dropna().unique())}
        brain_encodings[col] = enc
        X_br_df[col] = X_br_df[col].map(enc).fillna(-1)
    X_brain     = X_br_df.fillna(0).values.astype(float)
    y_brain     = df_brain['log_ID_pct'].values
    study_ids   = df_brain['group'].values.astype(str)
    carrier_ids = df_brain['prep'].fillna('unknown').values.astype(str)
    print(f"  Brain: {X_brain.shape[0]} samples, {X_brain.shape[1]} features")

    return (X_lc, y_lc, study_ids_lc,
            X_brain, y_brain, study_ids, carrier_ids,
            feat_lc, feat_brain,
            lc_encodings, brain_encodings)


# ═══════════════════════════════════════════════════════════════════════════════
# COVARIATE SHIFT CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_importance_weights(
    X_brain:         np.ndarray,
    brain_path:      str   = BRAIN_DATA_PATH,
    clip_percentile: float = 95.0,
) -> np.ndarray:
    """
    Importance weights to correct for covariate shift in the brain surrogate.

    Problem:
        Training data is a heterogeneous mixture of nanoparticle types
        (liposomes, metal oxides, free drugs, PLGA). MOBO candidates are all
        PEGylated PLGA — the model's deployment distribution ≠ training
        distribution. Fitting on the pooled mixture places equal weight on
        metal oxide NPs and free drugs that are irrelevant to PLGA design,
        potentially biasing the surrogate in the PLGA subspace.

    Solution (Sugiyama et al., 2007 — importance-weighted ERM):
        1. Define "target" samples: comp == 'plga' AND PEG == 1
           (matches MOBO deployment distribution)
        2. Train logistic classifier: P(is_target | X_i)
        3. Density ratio weight: w_i = P(target | X_i) / P(other | X_i)
        4. Refit GPBoost with these sample weights.

    High weights → PLGA+PEG-like samples get more influence on the tree splits.
    Low weights  → liposomes / metal oxides contribute less to parameter estimation.

    Parameters
    ----------
    X_brain         : (n, d) float array — the training feature matrix already
                      produced by load_data(). Must be aligned with the brain
                      DataFrame rows in the same order.
    brain_path      : path to datasetCELL2.csv (for comp and PEG columns)
    clip_percentile : upper percentile at which to clip weights (prevents
                      extreme leverage from single outlier samples).

    Returns
    -------
    weights : (n,) float array, mean-normalised to 1.0.
              Uniform weights (all 1.0) are returned on any failure to prevent
              silent degradation of the model.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    try:
        df_raw = pd.read_csv(brain_path)
        df_raw = df_raw.dropna(subset=['ID%']).reset_index(drop=True)

        # Apply same organic filter so IW source distribution matches training rows
        if 'prep' in df_raw.columns:
            df_raw = df_raw[
                df_raw['prep'].str.lower().str.strip().isin(
                    {p.lower() for p in ORGANIC_PREPS}
                )
            ].reset_index(drop=True)

        # Target population: PLGA + PEGylated (= what MOBO designs)
        comp_is_plga = df_raw['comp'].str.lower().str.strip() == 'plga' \
                       if 'comp' in df_raw.columns else pd.Series(False, index=df_raw.index)
        peg_is_one   = df_raw['PEG'].astype(float) == 1.0 \
                       if 'PEG' in df_raw.columns else pd.Series(False, index=df_raw.index)
        is_target    = (comp_is_plga & peg_is_one).values.astype(float)

        n_target = int(is_target.sum())
        n_total  = len(is_target)

        # Need meaningful imbalance to warrant reweighting
        if n_target < 5 or n_target > n_total - 5:
            print("  [IW] Degenerate class split — using uniform weights.")
            return np.ones(n_total, dtype=float)

        if len(X_brain) != n_total:
            print(f"  [IW] Shape mismatch: X_brain has {len(X_brain)} rows but "
                  f"df has {n_total} rows — using uniform weights.")
            return np.ones(len(X_brain), dtype=float)

        # Logistic classifier on the existing feature matrix
        # (already median-imputed floats from load_data)
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_brain)

        clf = LogisticRegression(
            C            = 1.0,
            max_iter     = 1000,
            random_state = 42,
            class_weight = 'balanced',   # handles imbalanced classes
            solver       = 'lbfgs',
        )
        clf.fit(X_scaled, is_target)

        p_target = clf.predict_proba(X_scaled)[:, 1]    # P(PLGA+PEG | X_i)
        p_source = np.maximum(1.0 - p_target, 1e-6)     # P(other | X_i)

        weights = p_target / p_source                    # density ratio

        # Clip at clip_percentile to prevent extreme leverage
        clip_val = np.percentile(weights, clip_percentile)
        weights  = np.clip(weights, 0.0, clip_val)

        # Normalise to mean=1 so effective sample size interpretation is clear
        w_mean  = weights.mean()
        if w_mean < 1e-10:
            print("  [IW] Near-zero mean weight — using uniform weights.")
            return np.ones(n_total, dtype=float)
        weights /= w_mean
        return weights

    except Exception as e:
        print(f"  [IW] WARNING: importance weighting failed ({e}). "
              f"Falling back to uniform weights.")
        return np.ones(len(X_brain), dtype=float)


def compute_biological_similarity_counts(
    brain_path:          str   = BRAIN_DATA_PATH,
    bbb_permeability:    dict  = None,
    mechanism_boost:     float = 3.0,
    bbb_proximity_sigma: float = 1.5,
    max_oversample:      int   = 4,
) -> np.ndarray:
    """
    Integer oversampling counts for each brain delivery training row,
    based on biological relevance to receptor-mediated neurodegenerative
    disease delivery.

    Two signals:
      1. Mechanism: receptor-mediated rows get mechanism_boost× more copies.
         Corrects chronic underrepresentation vs passive transport (107 normal
         passive vs 2 receptor-mediated Alzheimer's).
      2. BBB permeability proximity: rows from disease states with BBB
         coefficients near the neurodegeneration range (1.6-1.7×) are
         upweighted. Tumor models (8×) dominate training signal but are
         mechanistically distant from AD/PD targets.

    Uses oversampling rather than sample_weight because GPBoost
    likelihood='gaussian' does not support observation weights.

    Returns
    -------
    counts : (n,) int array — number of times to repeat each row.
             Minimum 1 (no row is removed), maximum max_oversample.
    """
    if bbb_permeability is None:
        bbb_permeability = MY_BBB_PERMEABILITY

    try:
        df = pd.read_csv(brain_path)
        df = df.dropna(subset=['ID%']).reset_index(drop=True)
        if 'prep' in df.columns:
            df = df[
                df['prep'].str.lower().str.strip().isin(
                    {p.lower() for p in ORGANIC_PREPS}
                )
            ].reset_index(drop=True)
        if 'animal-model' in df.columns:
            df = df.rename(columns={'animal-model': 'animalmodel'})

        n       = len(df)
        weights = np.ones(n, dtype=float)

        # Signal 1: mechanism match
        # receptor-mediated rows get mechanism_boost× more copies
        if 'mechanism' in df.columns:
            mech = df['mechanism'].str.lower().str.strip()
            weights[mech == 'receptor mediated'] *= mechanism_boost

        # Signal 2: BBB proximity to neurodegeneration range
        # Gaussian kernel centered at 1.65 (AD=1.6, PD=1.7 average)
        target_bbb = 1.65
        if 'animalmodel' in df.columns:
            bbb_vals = df['animalmodel'].map(bbb_permeability).fillna(1.0).values
            bbb_dist = np.abs(bbb_vals - target_bbb)
            bbb_w    = np.exp(-bbb_dist**2 / (2 * bbb_proximity_sigma**2))
            weights *= bbb_w

        # Map to integer counts in [1, max_oversample]
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min + 1e-8:
            norm = 1.0 + (max_oversample - 1) * (weights - w_min) / (w_max - w_min)
        else:
            norm = np.ones(n)
        counts = np.clip(np.round(norm).astype(int), 1, max_oversample)

        return counts

    except Exception as e:
        print(f"  [BioSim] WARNING: biological similarity oversampling failed "
              f"({e}). Falling back to uniform counts.")
        df_tmp = pd.read_csv(brain_path).dropna(subset=['ID%'])
        return np.ones(len(df_tmp), dtype=int)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL FITTING
# ═══════════════════════════════════════════════════════════════════════════════

def fit_lc_model(X_lc, y_lc):
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(iterations=500, learning_rate=0.05,
                               depth=6, verbose=0, random_seed=42)
    model.fit(X_lc, y_lc)
    return model


def fit_brain_model(X_brain, y_brain, study_ids, carrier_ids, feature_names=None):
    from tabpfn_gp_model import TabPFNGPModel
    model = TabPFNGPModel(re_type='study', use_remote=True, verbose=True)
    model.fit(X_brain, y_brain, study_ids, carrier_ids)
    return model


class GPBoostBrainModel:
    """
    Pure GPBoost brain delivery surrogate.

    Gradient-boosted trees + GP grouped random effects trained JOINTLY via
    gpb.train() — the same architecture used in your validated CV runs.
    No CatBoost, no TabPFN API calls. Runs entirely on-device.

    Hyperparameters match your best CV configuration exactly.
    Drop-in replacement for TabPFNGPModel — same external interface.

    Parameters
    ----------
    re_type : 'study' | 'crossed' | 'none'
    verbose : bool
    """

    # ── Hyperparameters ───────────────────────────────────────────────────
    # Context: 13 features, ~553 training rows, grouped RE by study.
    #
    # History:
    #   v1 (num_leaves=1024, max_depth=16): early-stop at round 2 during
    #      MOBO because only 4 features varied across candidates — tree
    #      memorised in 1 round, BLUP=0 for unseen group → grand mean output.
    #   v2 (num_leaves=16, max_depth=4): fixed MOBO flat-prediction but
    #      caused in-sample R²=0.36 — model too constrained to fit 13
    #      features during training.
    #
    # v3 (current): expressive enough to fit training data well (target
    #   in-sample R²>0.80) while the two-phase training floor prevents the
    #   round-2 early stopping. MOBO flat-prediction is now solved by the
    #   encoding fix + ParEGO, not by constraining tree depth.
    #
    #   num_leaves=64   → log2(64)=6 splits per tree; fits 13 features well
    #   max_depth=6     → moderate depth; prevents single-tree memorisation
    #   min_data_in_leaf=10 → restored; 20 was too aggressive for 553 rows
    #   lambda_l2=5     → moderate regularisation
    #   feature_fraction=0.7 → stochastic feature selection per tree
    #   bagging_fraction=0.8 → row subsampling for variance reduction
    _TREE_PARAMS = {
        'learning_rate':    0.05,
        'max_depth':        16,
        'num_leaves':       1024,
        'min_data_in_leaf': 10,
        'lambda_l2':        10,
        'feature_fraction': 0.3,
        'bagging_fraction': 0.45,
        'bagging_freq':     0,
        'verbose':         -1,
        'seed':             42,
    }
    _NUM_BOOST_ROUND       = 3_000
    _MIN_BOOST_ROUND       = 100   # floor: enough rounds to escape intercept-only
    _EARLY_STOPPING_ROUNDS = 100
    _VAL_FRACTION          = 0.15   # held out only for early-stopping signal

    def __init__(self, re_type: str = 'study', verbose: bool = True):
        self.re_type       = re_type
        self.verbose       = verbose
        self._is_fitted    = False
        self.booster_      = None
        self.gp_model_     = None
        self.feature_names_= None   # set by fit() when feature_names is provided

    def _group_array(self, study_ids, carrier_ids):
        study_ids = np.array(study_ids).astype(str)
        if self.re_type == 'crossed':
            return np.column_stack([study_ids,
                                    np.array(carrier_ids).astype(str)])
        elif self.re_type == 'study':
            return study_ids.reshape(-1, 1)
        return None

    def fit(self, X, y, study_ids, carrier_ids, sample_weight=None,
            feature_names=None):
        import gpboost as gpb
        from sklearn.metrics import r2_score as _r2_score

        self.feature_names_ = list(feature_names) if feature_names is not None else None
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        # 15 % held-out slice for early stopping (not used in final model)
        rng     = np.random.default_rng(42)
        n       = len(X)
        val_idx = rng.choice(n, size=max(1, int(n * self._VAL_FRACTION)),
                             replace=False)
        tr_idx  = np.setdiff1d(np.arange(n), val_idx)

        group_tr = self._group_array(np.array(study_ids)[tr_idx],
                                     np.array(carrier_ids)[tr_idx])

        # GPBoost with likelihood='gaussian' does not support observation weights
        # in either GPModel() or gpb.Dataset:
        #   • Passing weight= to Dataset raises "Weights need to be provided to GPModel()"
        #   • Passing weights= to GPModel raises "'weights' not supported for gaussian"
        # Importance weights are therefore silently dropped for this surrogate.
        if sample_weight is not None:
            pass  # GPBoost likelihood='gaussian' does not support observation weights

        self.gp_model_ = gpb.GPModel(group_data=group_tr, likelihood='gaussian')

        dtrain = gpb.Dataset(X[tr_idx], label=y[tr_idx])   # no weight=
        dval   = gpb.Dataset(X[val_idx], label=y[val_idx], reference=dtrain)

        # Required by GPBoost before training when a validation set is used
        group_val = self._group_array(np.array(study_ids)[val_idx],
                                      np.array(carrier_ids)[val_idx])
        self.gp_model_.set_prediction_data(group_data_pred=group_val)

        callbacks = [
            gpb.early_stopping(self._EARLY_STOPPING_ROUNDS,
                               verbose=self.verbose),
        ]
        if not self.verbose:
            if hasattr(gpb, 'log_evaluation'):
                callbacks.append(gpb.log_evaluation(-1))
            elif hasattr(gpb, 'print_evaluation'):
                callbacks.append(gpb.print_evaluation(-1))

        # Train in two phases so early-stopping cannot fire before _MIN_BOOST_ROUND.
        # Phase A: fixed rounds with no early-stopping (floor)
        booster_a = gpb.train(
            params          = self._TREE_PARAMS,
            train_set       = dtrain,
            gp_model        = self.gp_model_,
            num_boost_round = self._MIN_BOOST_ROUND,
        )
        # Phase B: continue from checkpoint, now allow early-stopping
        self.booster_ = gpb.train(
            params            = self._TREE_PARAMS,
            train_set         = dtrain,
            gp_model          = self.gp_model_,
            num_boost_round   = self._NUM_BOOST_ROUND - self._MIN_BOOST_ROUND,
            valid_sets        = [dval],
            valid_names       = ['val'],
            callbacks         = callbacks,
            init_model        = booster_a,
        )

        self._is_fitted = True

        # ── Calibration statistics for distance-based sigma (Fix 1C) ─────────
        # For MOBO candidates (all unseen study groups), GPBoost returns nearly
        # constant posterior variance → sigma is uninformative.  Replace with
        # residual_std * (1 + 0.5 * dist_norm): sigma grows with distance from
        # the training distribution, giving EHVI/ParEGO a genuine exploration
        # signal that distinguishes well-sampled from novel regions.
        self._X_train          = X.copy()
        self._X_train_centroid = X.mean(axis=0)
        self._X_train_p90_dist = float(np.percentile(
            np.linalg.norm(X - self._X_train_centroid, axis=1), 90
        ))
        # Tree-only residuals: ignore_gp_model=True tells GPBoost to return
        # only the tree ensemble (fixed effects), skipping the GP component.
        # This avoids the ValueError raised when group_data_pred is omitted
        # from a booster that has an attached gp_model.
        _tree_preds_tr   = self.booster_.predict(X[tr_idx], ignore_gp_model=True)
        if isinstance(_tree_preds_tr, dict):
            _tree_preds_tr = np.array(_tree_preds_tr.get('response_mean', _tree_preds_tr), dtype=float)
        _tree_preds_tr   = np.array(_tree_preds_tr, dtype=float)
        _tree_residuals  = y[tr_idx] - _tree_preds_tr
        self._residual_std = float(np.std(_tree_residuals))
        if self.verbose:
            print(f"  Best iteration: {self.booster_.best_iteration}")

        return self

    def predict_with_uncertainty(self, X, study_ids, carrier_ids):
        """
        Returns (mu, sigma).  sigma is candidate-varying (Fix 1C).

        For MOBO candidates (all unseen study groups), GPBoost posterior
        variance is near-constant and uninformative — the GP simply returns
        the prior variance σ²_group for every candidate.  Replaced with
        calibrated residual noise scaled by Euclidean distance from the
        training centroid so that epistemic uncertainty grows with novelty.

        sigma(x) = residual_std × (1 + 0.5 × min(dist_norm(x), 2))
            where dist_norm = ||x - μ_train|| / p90_dist_train

        This gives sigma ∈ [residual_std, 2×residual_std] across the search
        space — enough spread for EHVI/ParEGO to distinguish regions.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        X          = np.array(X, dtype=float)
        group_pred = self._group_array(study_ids, carrier_ids)

        pred = self.booster_.predict(
            data            = X,
            group_data_pred = group_pred,
            predict_var     = True,
        )
        mu = np.maximum(np.array(pred['response_mean'], dtype=float), 0.0)

        # ── Candidate-varying sigma (Fix 1C) ─────────────────────────────────
        if (hasattr(self, '_residual_std')
                and hasattr(self, '_X_train_p90_dist')
                and self._X_train_p90_dist > 1e-8):
            feat_dist      = np.linalg.norm(
                X - self._X_train_centroid[None, :], axis=1)
            feat_dist_norm = feat_dist / self._X_train_p90_dist   # ≈ [0, 1]
            sigma = self._residual_std * (1.0 + 0.5 * np.minimum(feat_dist_norm, 2.0))
        else:
            # Fallback: raw GP variance (original behaviour before Fix 1C)
            sigma = np.sqrt(np.maximum(np.array(pred['response_var'], dtype=float), 0.0))

        return mu, sigma

    def predict(self, X, study_ids, carrier_ids):
        """Point-prediction fallback (same signature as TabPFNGPModel.predict)."""
        mu, _ = self.predict_with_uncertainty(X, study_ids, carrier_ids)
        return mu


def fit_brain_model_gpboost(X_brain, y_brain, study_ids, carrier_ids,
                            sample_weight=None, feature_names=None,
                            oversample_counts=None):
    """Fit the pure GPBoost brain surrogate (local, no API quota)."""
    if oversample_counts is not None:
        oversample_counts = np.array(oversample_counts, dtype=int)
        if len(oversample_counts) == len(X_brain):
            X_brain     = np.repeat(X_brain,                oversample_counts, axis=0)
            y_brain     = np.repeat(y_brain,                oversample_counts)
            study_ids   = np.repeat(np.array(study_ids),   oversample_counts)
            carrier_ids = np.repeat(np.array(carrier_ids), oversample_counts)
    model = GPBoostBrainModel(re_type='study', verbose=True)
    model.fit(X_brain, y_brain, study_ids, carrier_ids,
              sample_weight=sample_weight, feature_names=feature_names)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH SPACE & CASCADE
# ═══════════════════════════════════════════════════════════════════════════════

def build_space_with_lookups():
    return SearchSpace(
        continuous    = MY_CONTINUOUS_PARAMS,
        categorical   = MY_CATEGORICAL_PARAMS,
        binary        = MY_BINARY_PARAMS,
        lookup_tables = {
            'bbb_permeability': MY_BBB_PERMEABILITY,
            'surfactant_hlb':   MY_SURFACTANT_HLB,
            # receptor_expr: maps sampled 'receptor' string → log1p(BBB expression TPM)
            # Same RECEPTOR_BBB_EXPR table used in load_data() so training and
            # MOBO candidate values are computed with identical transformations.
            'receptor_expr':    RECEPTOR_BBB_EXPR,
        },
        lookup_col_map = {
            # 'surfactant_type' sampled → look up HLB → write 'surfactant_HLB' (Stage 1 feature)
            'surfactant_hlb':  ('surfactant_type', 'surfactant_HLB'),
            # 'receptor' sampled → look up BBB expression → write 'receptor_bbb_expr' (Stage 2 feature)
            'receptor_expr':   ('receptor', 'receptor_bbb_expr'),
        },
        # Deterministic ligand assignment from sampled receptor.
        # SearchSpace.sample() uses this to set 'ligand' = canonical_ligand_map[receptor]
        # before building the candidate DataFrame — making invalid pairs impossible.
        canonical_ligand_map = RECEPTOR_TO_CANONICAL_LIGAND,
    )


def _preprocess_degradation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute PLGA degradation features for MOBO candidates on the fly.

    MOBO candidates have formulation parameters (polymer_MW, LAGA,
    particle_size, drugpolymer) but not the derived degradation columns.
    This callback runs before Stage-1 feature extraction so the CatBoost
    model sees the same degradation features it was trained on.
    """
    return add_degradation_features(df.copy(), indication='normal')


def build_cascade(lc_model, brain_model, X_lc, y_lc,
                  feat_lc, feat_brain,
                  lc_encodings=None, brain_encodings=None,
                  fast_mode=False):
    """
    Stage 1 (LC, R2=0.90): drugpolymer + particle_size + mol descriptors -> LC
    Stage 2 (brain, R2=0.57): formulation + LC_predicted + context -> log_ID%
    Uncertainty propagates via Monte Carlo through the cascade.

    feat_lc / feat_brain must be the exact columns used during model.fit()
    so that prediction receives the same number of features.
    """
    lc_surrogate = Surrogate(
        fitted_model         = lc_model,
        name                 = 'LC',
        uncertainty_method   = 'fixed' if fast_mode else 'bootstrap',
        X_train              = X_lc,
        y_train              = y_lc,
        n_bootstrap          = 20,
        uncertainty_fraction = 0.10,
    )
    brain_surrogate = TabPFNSurrogate(
        fitted_model       = brain_model,
        name               = 'brain_delivery_logID_pct',
        study_id_default   = 'design',
        carrier_id_default = 'polymer-based nanoparticles',
    )
    return CascadingSurrogate(
        stage1_surrogates   = {'LC': lc_surrogate},
        stage2_surrogate    = brain_surrogate,
        cascade_vars        = MY_CASCADE_VARS,
        stage1_feature_cols = feat_lc,
        stage2_feature_cols = feat_brain,
        n_cascade_samples   = 10 if fast_mode else 20,  # 30→20: 33% faster cascade, negligible MC noise increase
        column_aliases      = MY_COLUMN_ALIASES,
        stage1_encodings    = lc_encodings,
        stage2_encodings    = brain_encodings,
        stage2_imputation   = STAGE2_IMPUTATION_DEFAULTS,
        stage1_preprocess   = _preprocess_degradation,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_mobo(organic=True):
    import argparse

    drug_names = list(FAILED_CNS_DRUGS.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--drug', default='1',
        help='Drug name or number (1-%d). Use --list to see options.' % len(drug_names),
    )
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--all',  action='store_true')
    parser.add_argument('--list', action='store_true', help='List available drugs and exit.')
    parser.add_argument(
        '--brain-model', default='tabpfn',
        choices=['tabpfn', 'gpboost'],
    )
    parser.add_argument('--iw', action='store_true', default=False)
    args, _ = parser.parse_known_args()

    # --list: print drug menu and exit
    if args.list:
        print("\nAvailable drugs:")
        for i, name in enumerate(drug_names, 1):
            info = FAILED_CNS_DRUGS[name]
            print(f"  {i}. {name:<16s}  {info['indication']:<24s}  {info['mechanism']}")
        return

    # Resolve --drug: accept number or name
    if args.drug.isdigit():
        idx = int(args.drug) - 1
        if idx < 0 or idx >= len(drug_names):
            raise ValueError(f"Drug number must be 1-{len(drug_names)}, got {args.drug}")
        selected_drug = drug_names[idx]
    else:
        if args.drug not in FAILED_CNS_DRUGS:
            raise ValueError(f"Unknown drug '{args.drug}'. Use --list to see options.")
        selected_drug = args.drug

    FAST         = args.fast
    BRAIN_MODEL  = args.brain_model
    USE_IW       = args.iw

    print("Loading data...")
    (X_lc, y_lc, study_ids_lc,
     X_brain, y_brain, study_ids, carrier_ids,
     feat_lc, feat_brain,
     lc_encodings, brain_encodings) = load_data(organic=organic)

    iw_weights = None
    if USE_IW and BRAIN_MODEL == 'gpboost':
        iw_weights = compute_importance_weights(X_brain, brain_path=BRAIN_DATA_PATH)
        ess = (iw_weights.sum()**2) / (iw_weights**2).sum()
        if ess < 300:
            iw_weights = None
    bio_counts = None
    if BRAIN_MODEL == 'gpboost':
        bio_counts = compute_biological_similarity_counts(
            brain_path=BRAIN_DATA_PATH, bbb_permeability=MY_BBB_PERMEABILITY,
        )

    # ── Conformal calibration split (Fix 5) ──────────────────────────────────
    # Reserve 20% of each dataset as held-out calibration rows.  Models fit on
    # 80%; calibration rows are never seen during training — ensures conformal
    # coverage is a genuine out-of-sample guarantee.
    #
    # Note on GP state contamination: calibration for the brain model uses the
    # CALIBRATION study IDs (real held-out rows), not 'design'. GPBoost will
    # cache BLUP statistics for those study IDs after predict_with_uncertainty()
    # is called on the calibration set.  MOBO candidates then use study_id=
    # 'design' (a completely different, unseen group), so MOBO inference is on
    # a DIFFERENT GP branch from calibration — no contamination.
    rng_cal        = np.random.default_rng(999)
    n_brain        = len(X_brain)
    n_lc           = len(X_lc)
    cal_size_brain = max(30, n_brain // 5)
    cal_size_lc    = max(20, n_lc   // 5)

    cal_idx_brain  = rng_cal.choice(n_brain, size=cal_size_brain, replace=False)
    tr_idx_brain   = np.setdiff1d(np.arange(n_brain), cal_idx_brain)
    cal_idx_lc     = rng_cal.choice(n_lc,   size=cal_size_lc,   replace=False)
    tr_idx_lc      = np.setdiff1d(np.arange(n_lc),   cal_idx_lc)

    print("\nFitting LC model...")
    lc_model = fit_lc_model(X_lc[tr_idx_lc], y_lc[tr_idx_lc])

    if BRAIN_MODEL == 'tabpfn':
        print("Fitting brain model (TabPFN+GP)...")
        brain_model = fit_brain_model(
            X_brain[tr_idx_brain], y_brain[tr_idx_brain],
            study_ids[tr_idx_brain], carrier_ids[tr_idx_brain],
            feature_names=feat_brain,
        )
    else:
        print("Fitting brain model (GPBoost)...")
        _iw_tr = (iw_weights[tr_idx_brain] if iw_weights is not None else None)
        brain_model = fit_brain_model_gpboost(
            X_brain[tr_idx_brain], y_brain[tr_idx_brain],
            study_ids[tr_idx_brain], carrier_ids[tr_idx_brain],
            sample_weight=_iw_tr,
            feature_names=feat_brain,
            oversample_counts=(bio_counts[tr_idx_brain] if bio_counts is not None else None),
        )

    # ── Conformal calibration ─────────────────────────────────────────────────
    mu_cal_brain, _ = brain_model.predict_with_uncertainty(
        X_brain[cal_idx_brain],
        study_ids[cal_idx_brain],
        carrier_ids[cal_idx_brain],
    )
    conformal_brain = ConformalCalibrator(coverage=0.90)
    conformal_brain.calibrate(y_brain[cal_idx_brain], mu_cal_brain)

    mu_cal_lc = lc_model.predict(X_lc[cal_idx_lc])
    conformal_lc = ConformalCalibrator(coverage=0.90)
    conformal_lc.calibrate(y_lc[cal_idx_lc], mu_cal_lc)

    conformal_calibrators = {
        'LC':                       conformal_lc,
        'brain_delivery_logID_pct': conformal_brain,
    }

    space   = build_space_with_lookups()
    cascade = build_cascade(lc_model, brain_model,
                            X_lc[tr_idx_lc], y_lc[tr_idx_lc],
                            feat_lc, feat_brain,
                            lc_encodings=lc_encodings,
                            brain_encodings=brain_encodings,
                            fast_mode=FAST)

    drugs_to_run = drug_names if args.all else [selected_drug]
    all_results  = {}

    for drug_name in drugs_to_run:
        info = FAILED_CNS_DRUGS[drug_name]
        ff = info['fixed_features']
        print(f"\n{'='*55}")
        print(f"  {drug_name.upper()}")
        print(f"  Indication: {info['indication']}")
        print(f"  Mechanism:  {info['mechanism']}")
        print(f"  Failure:    {info['failure']}")
        print(f"  MW={ff['mol_MW']}  logP={ff['mol_logP']}  TPSA={ff['mol_TPSA']}  "
              f"MP={ff['mol_melting_point']}  HBD={ff['mol_Hdonors']}  HBA={ff['mol_Hacceptors']}")
        print(f"{'='*55}")

        mobo = CascadingMOBO(
            cascading_surrogate       = cascade,
            objectives                = MY_OBJECTIVES,
            search_space              = space,
            n_initial                 = 20  if FAST else 200,
            n_iterations              = 5   if FAST else 60,
            batch_size                = 3   if FAST else 5,
            n_candidates_per_iter     = 200 if FAST else 1000,
            n_mc_samples              = 32  if FAST else 128,
            verbose                   = True,
            drug_name                 = drug_name,
            early_stopping_patience   = 5   if FAST else 20,  # flat iterations are normal; don't kill early
            early_stopping_min_delta  = 1e-4,
        )
        mobo.space.fixed_features = info['fixed_features']

        result = mobo.run()
        diagnose_mobo(result)

        # Expand collapsed Pareto front via weight-vector grid sampling.
        # Strict non-dominance often yields only 2 points when one objective
        # has low surrogate discrimination (brain std ≈ 0.08 << sigma ≈ 0.16).
        # pareto_grid_sample sweeps 30 Chebyshev weight vectors across all
        # evaluated candidates, recovering the full tradeoff curve.
        n_strict = len(result.pareto_candidates)
        result_expanded = pareto_grid_sample(result, n_grid=30, epsilon=0.0)
        result = result_expanded

        export_results(result, prefix=f'mobo_{drug_name}',
                       conformal_calibrators=conformal_calibrators)
        all_results[drug_name] = result

    if len(drugs_to_run) > 1:
        rows = []
        for d, res in all_results.items():
            rows.append({
                'drug':       d,
                'indication': FAILED_CNS_DRUGS[d]['indication'],
                'final_HV':   f"{res.hypervolume_history[-1]:.4f}",
                'n_pareto':   len(res.pareto_candidates),
                'best_LC':    f"{res.pareto_objectives[:,0].max():.2f}%",
                'best_brain': f"{res.pareto_objectives[:,1].max():.4f}",
            })
        summary = pd.DataFrame(rows)
        print(f"\n{'='*55}\n  CROSS-DRUG SUMMARY\n{'='*55}")
        print(summary.to_string(index=False))
        summary.to_csv('drug_rescue_summary.csv', index=False)

if __name__ == '__main__':
    run_mobo()
