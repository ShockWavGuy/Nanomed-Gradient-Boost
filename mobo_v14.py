"""
═══════════════════════════════════════════════════════════════════════════════
PLGA Nanoparticle MOBO Design Loop
═══════════════════════════════════════════════════════════════════════════════

Architecture:
    Surrogate 1: TabPFN + GP RE  → Brain delivery (%ID/g)    [maximize]
    Surrogate 2: CatBoost        → Encapsulation efficiency   [maximize]

    Acquisition: Expected Hypervolume Improvement (EHVI) via Monte Carlo
                 No BoTorch dependency — implemented directly using
                 surrogate predictive distributions.

    Output: Pareto-optimal PLGA formulations with:
            - Full synthesis parameters
            - Predicted objective values + confidence intervals
            - Pareto rank and hypervolume contribution

Formulation search space (PLGA-specific, literature-validated bounds):
    - PLGA MW          (kDa)
    - LA:GA ratio      (%)
    - Drug loading     (% w/w)
    - Surfactant conc  (% w/v PVA)
    - Solvent          (categorical: DCM, EtAc, Acetone)
    - Particle size    (nm target)
    - PEG fraction     (% w/w)
    - Targeting ligand (categorical: none, transferrin, RGD, folate)

Requirements:
    pip install catboost tabpfn-client gpboost scikit-learn numpy pandas scipy

═══════════════════════════════════════════════════════════════════════════════
"""

import gc
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.special import expit
from scipy.stats import norm

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Search Space Definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PLGASearchSpace:
    """
    Literature-validated PLGA formulation parameter bounds.
    Continuous params: sampled uniformly within bounds.
    Categorical params: sampled uniformly from options.

    """

    # Continuous parameters: (min, max)
    continuous: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'PLGA_MW_kDa':       (7.0,   100.0),   # Polymer MW
        'LA_GA_ratio':       (50.0,  85.0),     # Lactide fraction (%)
        'drug_loading_pct':  (1.0,   20.0),     # Drug content (% w/w)
        'surfactant_pct':    (0.1,   5.0),      # PVA concentration (% w/v)
        'particle_size_nm':  (80.0,  400.0),    # Target size
        'PEG_fraction_pct':  (0.0,   15.0),     # PEGylation (% w/w)
    })

    # Categorical parameters: list of options
    categorical: Dict[str, List] = field(default_factory=lambda: {
        'solvent':          ['DCM', 'EtAc', 'Acetone'],
        'targeting_ligand': ['none', 'transferrin', 'RGD', 'folate'],
    })

    def sample(self, n: int, seed: int = None) -> pd.DataFrame:
        """Sample n candidates from the search space."""
        rng = np.random.default_rng(seed)
        rows = {}

        for param, (lo, hi) in self.continuous.items():
            rows[param] = rng.uniform(lo, hi, n)

        for param, options in self.categorical.items():
            rows[param] = rng.choice(options, n)

        return pd.DataFrame(rows)

    @property
    def param_names(self) -> List[str]:
        return list(self.continuous.keys()) + list(self.categorical.keys())

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode categorical params as integers for model input.
        Returns numpy array with categoricals label-encoded.
        """
        encoded = df.copy()
        for param, options in self.categorical.items():
            option_map = {v: i for i, v in enumerate(options)}
            encoded[param] = encoded[param].map(option_map)
        return encoded[self.param_names].values.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Surrogate Wrappers
# ─────────────────────────────────────────────────────────────────────────────

class BrainDeliverySurrogate:
    """
    Wraps fitted TabPFNGPModel for brain delivery prediction.
    Provides (mean, std) for MOBO acquisition.
    """

    def __init__(
        self,
        fitted_model,            # Fitted TabPFNGPModel instance
        study_id_default: str = 'design',    # Pseudo study_id for candidates
        carrier_id_default: str = 'PLGA',    # Carrier type for candidates
    ):
        self.model = fitted_model
        self.study_id_default = study_id_default
        self.carrier_id_default = carrier_id_default

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (mean, std) for each candidate.

        Note: candidates are new formulations — they'll always be
        'unseen' studies, so RE correction = 0 (shrinkage applies).
        TabPFN fixed effects carry the full prediction load here.
        The GP uncertainty is still informative via marginal variance.
        """
        n = len(X)
        study_ids   = np.array([self.study_id_default] * n)
        carrier_ids = np.array([self.carrier_id_default] * n)

        try:
            mu, sigma = self.model.predict_with_uncertainty(
                X, study_ids, carrier_ids
            )
            # If sigma is all NaN (re_type='none'), use bootstrap fallback
            if np.all(np.isnan(sigma)):
                sigma = np.full(n, np.nanstd(mu) * 0.3)
        except Exception as e:
            print(f"  Warning: uncertainty prediction failed ({e}), "
                  f"using point predictions with fixed uncertainty")
            mu = self.model.predict(X, study_ids, carrier_ids)
            sigma = np.full(n, np.nanstd(mu) * 0.3)

        return mu, np.maximum(sigma, 1e-6)


class EncapsulationSurrogate:
    """
    Wraps fitted CatBoost model for encapsulation efficiency prediction.
    Provides (mean, std) via bootstrap ensemble or native uncertainty.
    """

    def __init__(
        self,
        fitted_model,            # Fitted CatBoostRegressor
        X_train: np.ndarray,     # Training features (for bootstrap fallback)
        y_train: np.ndarray,     # Training targets
        n_bootstrap: int = 30,   # Bootstrap samples for uncertainty
    ):
        self.model = fitted_model
        self.X_train = X_train
        self.y_train = y_train
        self.n_bootstrap = n_bootstrap
        self._bootstrap_models = None

    def _fit_bootstrap(self):
        """Fit bootstrap ensemble for uncertainty estimation."""
        from catboost import CatBoostRegressor
        rng = np.random.default_rng(42)
        self._bootstrap_models = []

        print("  Fitting CatBoost bootstrap ensemble for uncertainty...")
        for i in range(self.n_bootstrap):
            idx = rng.integers(0, len(self.X_train), len(self.X_train))
            m = CatBoostRegressor(
                iterations=self.model.get_param('iterations') or 300,
                verbose=0,
                random_seed=i,
            )
            m.fit(self.X_train[idx], self.y_train[idx])
            self._bootstrap_models.append(m)
        print(f"  Bootstrap ensemble ready ({self.n_bootstrap} models)")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) across bootstrap ensemble."""
        if self._bootstrap_models is None:
            self._fit_bootstrap()

        preds = np.array([m.predict(X) for m in self._bootstrap_models])
        return preds.mean(axis=0), np.maximum(preds.std(axis=0), 1e-6)


class SimpleSurrogate:
    """
    Minimal surrogate for any sklearn-compatible regressor.
    Uses prediction + fixed fractional uncertainty.
    Suitable when bootstrap is too slow.
    """

    def __init__(self, fitted_model, uncertainty_fraction: float = 0.15):
        """uncertainty_fraction: std = uncertainty_fraction * |mean|"""
        self.model = fitted_model
        self.uncertainty_fraction = uncertainty_fraction

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = self.model.predict(X)
        sigma = np.maximum(np.abs(mu) * self.uncertainty_fraction, 1e-6)
        return mu, sigma


# ─────────────────────────────────────────────────────────────────────────────
# Conformal Prediction
# ─────────────────────────────────────────────────────────────────────────────

class ConformalCalibrator:
    """
    Split conformal prediction for regression (Inductive CP, Papadopoulos 2002).

    Wraps any point-predicting surrogate and replaces bootstrap/GP uncertainty
    with prediction intervals that carry a GUARANTEED finite-sample marginal
    coverage property:

        P(y_new ∈ [ŷ_new − q̂, ŷ_new + q̂]) ≥ 1 − α

    for any sample size and any model, with no distributional assumptions
    beyond exchangeability of calibration and test residuals.

    This is fundamentally stronger than bootstrap or GP variance, which only
    have coverage guarantees asymptotically and under model specification.

    Parameters
    ----------
    coverage : float
        Desired marginal coverage (e.g. 0.90 → 90% of true values fall inside).
        Must be in (0, 1).

    Usage
    -----
    cal = ConformalCalibrator(coverage=0.90)
    cal.calibrate(y_cal, mu_cal)           # fit on held-out calibration split
    lo, hi = cal.predict_interval(mu_new)  # apply to new predictions
    print(cal.summary())

    Notes
    -----
    Conformal quantile formula (Angelopoulos & Bates 2022):
        level = ceil((n+1)(1−α)) / n
        q̂    = quantile(|y_i − ŷ_i|, level)
    The +1 correction gives exact (not asymptotic) coverage.
    Intervals are symmetric (constant width) — a consequence of using the
    absolute residual as nonconformity score. For heteroscedastic data
    a normalised score (dividing by predicted sigma) would give adaptive
    widths; this simpler version matches the standard implementation
    in pharmaceutical QSAR literature.
    """

    def __init__(self, coverage: float = 0.90):
        if not 0 < coverage < 1:
            raise ValueError(f"coverage must be in (0, 1), got {coverage}")
        self.coverage  = coverage
        self._q_hat    = None
        self._n_cal    = None
        self._residuals = None   # kept for diagnostics

    def calibrate(
        self,
        y_cal:  np.ndarray,
        mu_cal: np.ndarray,
    ) -> 'ConformalCalibrator':
        """
        Compute the conformal quantile q̂ from calibration residuals.

        Parameters
        ----------
        y_cal  : (n,) true target values on held-out calibration set
        mu_cal : (n,) surrogate point predictions on the same set

        The calibration set MUST NOT have been seen during model training —
        use a GroupKFold-compatible hold-out (one study group, or a random
        20% split stratified by study) for valid coverage on new studies.
        """
        y_cal  = np.asarray(y_cal,  dtype=float)
        mu_cal = np.asarray(mu_cal, dtype=float)
        if len(y_cal) != len(mu_cal):
            raise ValueError("y_cal and mu_cal must have the same length")

        scores = np.abs(y_cal - mu_cal)   # nonconformity scores
        n = len(scores)
        self._n_cal    = n
        self._residuals = scores

        # Conformal quantile: the (1−α)(n+1)/n level of the score distribution.
        # This exact formula ensures P(coverage) ≥ 1−α for finite n.
        alpha   = 1.0 - self.coverage
        level   = min(np.ceil((1.0 - alpha) * (n + 1)) / n, 1.0)
        self._q_hat = float(np.quantile(scores, level))
        return self

    def predict_interval(
        self,
        mu: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute calibrated prediction intervals for new predictions.

        Returns (lower, upper) with guaranteed ≥ coverage marginal probability.
        """
        if self._q_hat is None:
            raise RuntimeError("Call calibrate() before predict_interval()")
        mu = np.asarray(mu, dtype=float)
        return mu - self._q_hat, mu + self._q_hat

    @property
    def half_width(self) -> float:
        """Symmetric ± half-width of the intervals (same as q̂)."""
        if self._q_hat is None:
            raise RuntimeError("Not calibrated yet.")
        return self._q_hat

    def coverage_check(self, y_test: np.ndarray, mu_test: np.ndarray) -> float:
        """
        Empirical coverage on a separate test set (optional validation).
        Should be ≥ self.coverage when calibration and test are exchangeable.
        """
        lo, hi = self.predict_interval(mu_test)
        return float(np.mean((y_test >= lo) & (y_test <= hi)))

    def summary(self) -> str:
        if self._q_hat is None:
            return "ConformalCalibrator (not calibrated)"
        med_r = float(np.median(self._residuals)) if self._residuals is not None else float('nan')
        return (
            f"ConformalCalibrator("
            f"coverage={self.coverage:.0%}, "
            f"n_cal={self._n_cal}, "
            f"q̂={self._q_hat:.4f}, "
            f"median_residual={med_r:.4f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Receptor Expression Atlas
# ─────────────────────────────────────────────────────────────────────────────

# Disease-state receptor expression multipliers derived from mouse brain
# endothelial transcriptomics (healthy baseline normalised to 1.0).
#
# Source: single-study mouse brain endothelial dataset (healthy + 4 disease states)
# Method: log2FC columns converted to linear multipliers (2^log2FC), capped at 8×.
#   - Multi-subunit receptors: integrin (min of Itgav/Itgb3 — obligate heterodimer)
#   - Parallel pathways:       LDL family (mean of Ldlr/Lrp1 — independent receptors)
#   - Zero-expressed genes:    Folr1, Slc5a7 → fixed at 1.0 (no BBB expression)
#
# Disease mapping to available conditions:
#   Alzheimer's disease → EAE chronic   (shared chronic neuroinflammatory mechanism)
#   brain tumor         → TBI acute     (shared acute BBB disruption)
#   stroke              → stroke acute  (direct match)
#   epilepsy            → epilepsy chronic (maintenance therapy context)
#
# Applied in log-space: mu_log1p += log(multiplier)
# Equivalent to multiplicative scaling of linear %ID/g predictions.

RECEPTOR_EXPRESSION_ATLAS: Dict[str, Dict[str, float]] = {
    'transferrin receptor': {
        'epilepsy':            0.745,
        "Alzheimer's disease": 0.499,
        'stroke':              1.157,
        'brain tumor':         0.461,
        'healthy':             1.000,
    },
    'hexose transporter': {
        'epilepsy':            0.725,
        "Alzheimer's disease": 0.755,
        'stroke':              0.865,
        'brain tumor':         0.793,
        'healthy':             1.000,
    },
    'low density lipoprotein-related receptor': {
        'epilepsy':            1.018,
        "Alzheimer's disease": 1.576,
        'stroke':              0.734,
        'brain tumor':         1.342,
        'healthy':             1.000,
    },
    'integrin': {
        'epilepsy':            0.759,
        "Alzheimer's disease": 1.393,
        'stroke':              0.769,
        'brain tumor':         1.569,
        'healthy':             1.000,
    },
    'folic acid receptor': {        # Folr1 not expressed at BBB — no modulation
        'epilepsy':            1.000,
        "Alzheimer's disease": 1.000,
        'stroke':              1.000,
        'brain tumor':         1.000,
        'healthy':             1.000,
    },
    'insulin receptor': {
        'epilepsy':            0.853,
        "Alzheimer's disease": 0.769,
        'stroke':              0.791,
        'brain tumor':         1.157,
        'healthy':             1.000,
    },
    'amino acid transporter': {
        'epilepsy':            0.914,
        "Alzheimer's disease": 0.558,
        'stroke':              0.799,
        'brain tumor':         0.588,
        'healthy':             1.000,
    },
    'monocarboxylic acid transporter': {
        'epilepsy':            0.648,
        "Alzheimer's disease": 0.794,
        'stroke':              0.874,
        'brain tumor':         1.112,
        'healthy':             1.000,
    },
    'choline transporter': {        # Slc5a7 not expressed at BBB — no modulation
        'epilepsy':            1.000,
        "Alzheimer's disease": 1.000,
        'stroke':              1.000,
        'brain tumor':         1.000,
        'healthy':             1.000,
    },
    'nucleoside transporter': {
        'epilepsy':            0.916,
        "Alzheimer's disease": 1.062,
        'stroke':              1.065,
        'brain tumor':         1.237,
        'healthy':             1.000,
    },
    'na+-coupled carnitine transporter 2': {
        'epilepsy':            1.077,
        "Alzheimer's disease": 1.125,
        'stroke':              0.768,
        'brain tumor':         0.750,
        'healthy':             1.000,
    },
    'sodium dependent-vitamin c transporters': {
        'epilepsy':            0.995,
        "Alzheimer's disease": 1.237,
        'stroke':              0.758,
        'brain tumor':         1.722,
        'healthy':             1.000,
},
}
# Fallback for unlabelled / heterogeneous receptor types
_ATLAS_DEFAULT: Dict[str, float] = {d: 1.0 for d in [
    'epilepsy', "Alzheimer's disease", 'stroke', 'brain tumor', 'healthy'
]}

def get_atlas_log_offset(
    receptor_type: str,
    disease_state: str,
) -> float:
    """
    Return log(expression_multiplier) for (receptor, disease) pair.
    Applied additively to log1p(%ID/g) predictions.
    Falls back to 0.0 (no correction) for unknown receptor or disease types.
    """
    rec_lower = str(receptor_type).lower().strip()
    # Normalise common aliases from datasetCELL2.csv receptor-type column
    _ALIAS = {
        # Hexose transporters
        'glut-1':                                       'hexose transporter',
        'glut1':                                        'hexose transporter',
        'glucose transporter':                          'hexose transporter',
        'glucose transporter-1':                        'hexose transporter',

        # Transferrin receptor variants
        'transferrin receptor':                         'transferrin receptor',
        'tfr':                                          'transferrin receptor',
        '转铁蛋白受体':                                     'transferrin receptor',
        'transferrin receptor+dopamine receptor':       'transferrin receptor',

        # LDL family (all variants)
        'low density lipoprotein-related receptor':     'low density lipoprotein-related receptor',
        'low density liproprotein-related receptor':    'low density lipoprotein-related receptor',
        'low density lipoprotein receptor-related protein': 'low density lipoprotein-related receptor',
        'ldlr':                                         'low density lipoprotein-related receptor',
        'lrp1':                                         'low density lipoprotein-related receptor',
        'lrp-1':                                        'low density lipoprotein-related receptor',
        'apolipoprotein e receptor':                    'low density lipoprotein-related receptor',
        'apoe3':                                        'low density lipoprotein-related receptor',
        'very low-density lipoprotein (vldl) receptor': 'low density lipoprotein-related receptor',
        'lactoferrin recepter':                         'low density lipoprotein-related receptor',
        'lactoferrin receptor':                         'low density lipoprotein-related receptor',
        '乳铁蛋白受体':                                     'low density lipoprotein-related receptor',

        # Integrin variants
        'integrin':                                     'integrin',
        'αvβ1':                                         'integrin',
        'αvβ3':                                         'integrin',
        '𝛂v𝛃3':                                         'integrin',

        # Folate
        'folic acid receptor':                          'folic acid receptor',
        'folate receptor':                              'folic acid receptor',

        # Amino acid / transport
        'amino acid transporter':                       'amino acid transporter',
        'l-amino acid transporter':                     'amino acid transporter',

        # Monocarboxylic acid (trailing space variant)
        'monocarboxylic acid transporter':              'monocarboxylic acid transporter',
        'monocarboxylic acid transporter ':             'monocarboxylic acid transporter',

        # Other transporters
        'insulin receptor':                             'insulin receptor',
        'choline transporter':                          'choline transporter',
        'nucleoside transporter':                       'nucleoside transporter',
        'na+-coupled carnitine transporter 2':          'na+-coupled carnitine transporter 2',
        'sodium dependent-vitamin c transporters':      'sodium dependent-vitamin c transporters',

        # Bin to others — too ambiguous or non-BBB
        'vcam-1':                                       'others',
        'vcam1':                                        'others',
        'mannose receptor [cluster of differentiation (cd) 206]': 'others',
        'nicotinicreceptor':                            'others',
        'nicotinic acetylcholine receptor (nachr)':     'others',
        'c-type lectin receptors and integrin lfa-1/icam-1': 'others',
        'both integrin αvβ3 and neuropilin-1 receptors': 'others',
        'aβ42':                                         'others',
        'combo':                                        'others',
        'cell-penetrating peptides (cpps)':             'others',
        'il-13rα2 endocytosis':                         'others',
        'neural cell adhesion molecule (ncam);p75 neurotrophin receptor (p75ntr);\nnicotinic acetylcholine receptors (nachrs)': 'others',
}
    canonical = _ALIAS.get(rec_lower, None)
    rec_dict  = RECEPTOR_EXPRESSION_ATLAS.get(canonical, _ATLAS_DEFAULT)

    # Normalise disease state — map disease strings to atlas keys
    _DISEASE_MAP = {
        "alzheimer's disease":  "Alzheimer's disease",
        'alzheimers disease':   "Alzheimer's disease",
        'alzheimer':            "Alzheimer's disease",
        'brain tumor':          'brain tumor',
        'tumor':                'brain tumor',
        'glioma':               'brain tumor',
        'stroke':               'stroke',
        'ischemic stroke':      'stroke',
        'ischemic stroke/reperfusion': 'stroke',
        'stroke(ischemic stroke, hemorrhagic stroke)': 'stroke',
        'epilepsy':             'epilepsy',
        'normal':               'healthy',
        'healthy':              'healthy',
        "parkinson's disease":  'healthy',   # no Parkinson's condition → baseline
        'encephalomyelitis':    "Alzheimer's disease",  # inflammatory → EAE proxy
        'others':               'healthy',
        'unknown':              'healthy',
    }
    dis_lower  = str(disease_state).lower().strip()
    atlas_key  = _DISEASE_MAP.get(dis_lower, 'healthy')
    multiplier = rec_dict.get(atlas_key, 1.0)
    return float(np.log(max(multiplier, 1e-6)))


# ─────────────────────────────────────────────────────────────────────────────
# Pareto Utilities
# ─────────────────────────────────────────────────────────────────────────────

def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """
    Find Pareto-efficient points (minimization convention).
    Pass negated objectives for maximization.

    Returns boolean mask — True = Pareto-efficient.
    """
    n = len(costs)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if is_efficient[i]:
            # Point i is dominated if any other point is better in all objectives
            dominated = np.all(costs[i] <= costs, axis=1) & \
                        np.any(costs[i] < costs, axis=1)
            dominated[i] = False
            is_efficient[is_efficient] = ~dominated[is_efficient]
    return is_efficient


def pareto_mask_2obj(Y: np.ndarray) -> np.ndarray:
    """
    O(n log n) Pareto-efficient mask for exactly 2 maximization objectives.

    Replaces is_pareto_efficient(-Y) for the common 2-objective case.
    Algorithm: sort by obj-0 descending; sweep and track the running max of
    obj-1 — a point is non-dominated iff its obj-1 exceeds every previous
    (higher obj-0) point's obj-1.

    Returns boolean mask (True = Pareto-efficient), same convention as
    is_pareto_efficient(-Y) called with maximisation negation.
    """
    n = len(Y)
    if n == 0:
        return np.zeros(0, dtype=bool)
    # Sort by obj-0 descending; ties broken by obj-1 descending
    order   = np.lexsort((-Y[:, 1], -Y[:, 0]))
    mask    = np.zeros(n, dtype=bool)
    best_y1 = -np.inf
    for idx in order:
        if Y[idx, 1] > best_y1:
            mask[idx] = True
            best_y1   = Y[idx, 1]
    return mask


def hypervolume_2d(pareto_points: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Compute 2D hypervolume indicator (maximization convention).
    Sort by first objective descending, accumulate rectangles.
    """
    if len(pareto_points) == 0:
        return 0.0
    pts = pareto_points[pareto_points[:, 0].argsort()[::-1]]
    hv = 0.0
    prev_y = ref_point[1]
    for pt in pts:
        if pt[0] > ref_point[0] and pt[1] >= prev_y - 1e-10:
            hv += (pt[0] - ref_point[0]) * (pt[1] - prev_y)
            prev_y = pt[1]
    return hv


def expected_hypervolume_improvement(
    mu: np.ndarray,           # (n_candidates, n_objectives)
    sigma: np.ndarray,        # (n_candidates, n_objectives)
    pareto_front: np.ndarray, # (n_pareto, n_objectives) current Pareto front
    ref_point: np.ndarray,    # (n_objectives,) reference point (below all)
    n_samples: int = 512,
) -> np.ndarray:
    """
    Monte Carlo Expected Hypervolume Improvement — vectorized.

    Speedup over the naive nested loop:
      1. All (n_candidates × n_samples) MC draws are batched in one NumPy call.
      2. Dominated-sample pre-filter: a single NumPy broadcast determines which
         samples are dominated by the *fixed* current Pareto front. Dominated
         samples contribute zero HVI — no Python work needed for them.
      3. For the (typically 5–20%) non-dominated samples, HVI is computed with
         an O(p) in-place filter instead of a full O(p²) is_pareto_efficient()
         call: just mask out front points that the new sample strictly dominates.
      4. Memory guard: if n_pareto × n_candidates × n_samples > 40 M elements,
         the dominated check runs in candidate-chunks of ≤200 to stay bounded.

    Returns EHVI score per candidate (higher = better to evaluate next).
    """
    n_candidates, n_obj = mu.shape
    ehvi = np.zeros(n_candidates)

    current_hv = hypervolume_2d(pareto_front, ref_point) \
        if len(pareto_front) > 0 else 0.0

    # ── Draw all MC samples at once: (n_candidates, n_samples, n_obj) ─────────
    all_samples = (mu[:, None, :] +
                   sigma[:, None, :] * np.random.standard_normal(
                       (n_candidates, n_samples, n_obj)))

    # ── Empty-front short-circuit ──────────────────────────────────────────────
    if len(pareto_front) == 0:
        for i in range(n_candidates):
            hvs = np.array([hypervolume_2d(all_samples[i, j:j+1], ref_point)
                            for j in range(n_samples)])
            ehvi[i] = np.maximum(hvs - current_hv, 0.0).mean()
        return ehvi

    pf = pareto_front  # (p, n_obj)
    p  = len(pf)

    # ── Vectorized dominance check ─────────────────────────────────────────────
    # is_dominated[i, j] ↔ sample j of candidate i is strictly dominated by
    # at least one point in the *fixed* current Pareto front.
    # Chunk candidates if the broadcast tensor would exceed ~40 M elements.
    chunk = max(1, int(40_000_000 // (n_samples * p * n_obj)))
    is_dominated = np.empty((n_candidates, n_samples), dtype=bool)

    for start in range(0, n_candidates, chunk):
        end   = min(start + chunk, n_candidates)
        s_blk = all_samples[start:end, :, None, :]  # (blk, mc, 1, n_obj)
        pf_   = pf[None, None, :, :]                 # (1,   1,  p, n_obj)
        weakly   = np.all(pf_ >= s_blk, axis=-1)    # (blk, mc, p)
        strictly = np.any(pf_ >  s_blk, axis=-1)    # (blk, mc, p)
        is_dominated[start:end] = np.any(weakly & strictly, axis=-1)

    # ── HVI only for non-dominated samples ────────────────────────────────────
    for i in range(n_candidates):
        non_dom = np.where(~is_dominated[i])[0]
        if len(non_dom) == 0:
            continue  # ehvi[i] stays 0
        improvements = np.zeros(n_samples)
        for j in non_dom:
            s = all_samples[i, j]  # (n_obj,)
            # O(p) augmented front: remove front points that s strictly dominates
            s_dom_pf = (np.all(s[None, :] >= pf, axis=-1) &
                        np.any(s[None, :] >  pf, axis=-1))
            aug_pf   = np.vstack([pf[~s_dom_pf], s[None, :]])
            improvements[j] = max(hypervolume_2d(aug_pf, ref_point) - current_hv, 0.0)
        ehvi[i] = improvements.mean()

    return ehvi


# ─────────────────────────────────────────────────────────────────────────────
# MOBO Loop
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MOBOResult:
    """Container for MOBO optimization results."""
    pareto_candidates: pd.DataFrame      # Formulation params
    pareto_objectives: np.ndarray        # Predicted objective values
    pareto_uncertainties: np.ndarray     # Std per objective
    hypervolume_history: List[float]     # HV per iteration
    all_candidates: pd.DataFrame         # Full evaluated set
    all_objectives: np.ndarray
    all_uncertainties: np.ndarray        # Std for full evaluated set
    ref_point: np.ndarray
    objective_names: List[str]


class PLGAMOBOLoop:
    """
    Multi-Objective Bayesian Optimization for PLGA nanoparticle design.

    Iteratively:
        1. Sample large random candidate set from search space
        2. Predict objectives + uncertainty via surrogates
        3. Compute EHVI acquisition for each candidate
        4. Select top-k candidates (batch acquisition)
        5. Add to observed set, update Pareto front
        6. Repeat

    Since this is a computational (not wet-lab) loop, 'evaluation'
    is surrogate prediction — no real experiments required. The loop
    explores the formulation space to build a Pareto front of
    optimized designs under model uncertainty.

    Parameters
    ----------
    surrogates : dict
        {'objective_name': surrogate_instance, ...}
        Each surrogate must implement .predict(X) → (mu, sigma)
    objectives : dict
        {'objective_name': 'maximize' | 'minimize'}
    search_space : PLGASearchSpace
    ref_point : array (n_objectives,)
        Hypervolume reference point — set below worst expected values.
        e.g. [0.0, 0.0] for brain delivery and EE both maximized.
    n_initial : int
        Initial random designs before EHVI kicks in.
    n_iterations : int
        Number of MOBO iterations after initialization.
    batch_size : int
        Candidates selected per iteration (q-batch acquisition).
    n_candidates_per_iter : int
        Random candidates evaluated per iteration for EHVI.
    n_mc_samples : int
        Monte Carlo samples for EHVI computation.
    """

    def __init__(
        self,
        surrogates: Dict,
        objectives: Dict[str, str],
        search_space: PLGASearchSpace,
        ref_point: Optional[np.ndarray] = None,
        n_initial: int = 50,
        n_iterations: int = 30,
        batch_size: int = 5,
        n_candidates_per_iter: int = 2000,
        n_mc_samples: int = 256,
        seed: int = 42,
        verbose: bool = True,
    ):
        if set(surrogates.keys()) != set(objectives.keys()):
            raise ValueError("surrogates and objectives must have the same keys")

        self.surrogates = surrogates
        self.objectives = objectives
        self.search_space = search_space
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.n_candidates_per_iter = n_candidates_per_iter
        self.n_mc_samples = n_mc_samples
        self.seed = seed
        self.verbose = verbose

        self.objective_names = list(objectives.keys())
        self.n_obj = len(self.objective_names)

        # Sign convention: EHVI always maximizes
        # For 'minimize' objectives, negate predictions
        self._signs = np.array([
            1.0 if objectives[k] == 'maximize' else -1.0
            for k in self.objective_names
        ])

        # Reference point for hypervolume (in maximization space)
        if ref_point is None:
            self.ref_point = np.zeros(self.n_obj)
        else:
            self.ref_point = np.array(ref_point) * self._signs

        # Storage
        self._X_observed = []       # Encoded feature arrays
        self._df_observed = []      # Raw dataframes (human-readable)
        self._Y_observed = []       # Objective arrays (maximization space)
        self._sigma_observed = []   # Uncertainty arrays
        self._hv_history = []
        self._pareto_mask = None

    def _predict_all(self, X_encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict all objectives for candidate set.
        Returns (mu, sigma) each shape (n_candidates, n_objectives),
        in maximization convention (minimization objectives negated).
        """
        mu_list, sigma_list = [], []

        for i, name in enumerate(self.objective_names):
            mu_i, sigma_i = self.surrogates[name].predict(X_encoded)
            mu_list.append(mu_i * self._signs[i])
            sigma_list.append(sigma_i)  # std is always positive

        return np.column_stack(mu_list), np.column_stack(sigma_list)

    def _update_pareto(self):
        """Recompute Pareto front from all observed points."""
        Y = np.vstack(self._Y_observed)
        self._pareto_mask = is_pareto_efficient(-Y)  # negate for maximization check

    def _current_pareto_front(self) -> np.ndarray:
        if len(self._Y_observed) == 0:
            return np.empty((0, self.n_obj))
        Y = np.vstack(self._Y_observed)
        return Y[self._pareto_mask]

    def _current_hypervolume(self) -> float:
        pf = self._current_pareto_front()
        if len(pf) == 0:
            return 0.0
        return hypervolume_2d(pf, self.ref_point)

    def run(self) -> MOBOResult:
        """
        Execute the full MOBO loop.

        Phase 1: Random initialization (n_initial designs)
        Phase 2: EHVI-guided optimization (n_iterations × batch_size)
        """
        rng = np.random.default_rng(self.seed)

        if self.verbose:
            print("═" * 60)
            print("PLGA MOBO DESIGN LOOP")
            print(f"Objectives: {self.objectives}")
            print(f"Iterations: {self.n_iterations}  |  "
                  f"Batch size: {self.batch_size}  |  "
                  f"Initial: {self.n_initial}")
            print("═" * 60)

        # ── Phase 1: Random initialization ────────────────────────────
        if self.verbose:
            print(f"\n[Phase 1] Random initialization ({self.n_initial} designs)")

        init_df = self.search_space.sample(self.n_initial, seed=self.seed)
        init_X  = self.search_space.encode(init_df)
        init_mu, init_sigma = self._predict_all(init_X)

        self._X_observed.append(init_X)
        self._df_observed.append(init_df)
        self._Y_observed.append(init_mu)
        self._sigma_observed.append(init_sigma)

        self._update_pareto()
        hv0 = self._current_hypervolume()
        self._hv_history.append(hv0)

        if self.verbose:
            pf = self._current_pareto_front()
            print(f"  Initial Pareto front: {len(pf)} points")
            print(f"  Initial hypervolume:  {hv0:.4f}")

        # ── Phase 2: EHVI-guided iterations ───────────────────────────
        for iteration in range(1, self.n_iterations + 1):
            if self.verbose:
                print(f"\n[Iteration {iteration}/{self.n_iterations}]")

            # Sample large candidate pool
            candidates_df = self.search_space.sample(
                self.n_candidates_per_iter,
                seed=int(rng.integers(0, 1e6)),
            )
            candidates_X = self.search_space.encode(candidates_df)

            # Predict objectives + uncertainty
            mu, sigma = self._predict_all(candidates_X)

            # Compute EHVI for each candidate
            pareto_front = self._current_pareto_front()
            ehvi = expected_hypervolume_improvement(
                mu=mu,
                sigma=sigma,
                pareto_front=pareto_front,
                ref_point=self.ref_point,
                n_samples=self.n_mc_samples,
            )

            # Select top batch_size candidates
            # Diversity-enforced batch selection:
            # Pure greedy top-k by EHVI causes all batch candidates to cluster
            # near the same region (typically best-LC), preventing Pareto spread.
            # Fix: greedy k-medoid — each new candidate must be at least
            # `min_spread` apart from already-selected ones in objective space.
            # Falls back to pure EHVI if spread constraint is unsatisfiable.
            sorted_idx = np.argsort(ehvi)[::-1]
            selected   = [sorted_idx[0]]
            # Compute target spread as 10% of the current Pareto front range
            pf_now   = self._current_pareto_front()
            if len(pf_now) >= 2:
                obj_range  = pf_now.max(0) - pf_now.min(0)
                min_spread = 0.10 * np.linalg.norm(obj_range)
            else:
                min_spread = 0.0  # no diversity constraint until Pareto has spread
            for idx in sorted_idx[1:]:
                if len(selected) >= self.batch_size:
                    break
                if min_spread > 0:
                    Y_sel = mu[selected]               # (k, n_obj)
                    dists = np.linalg.norm(mu[idx] - Y_sel, axis=1)
                    if dists.min() < min_spread:
                        continue                        # too close — skip
                selected.append(idx)
            # Fallback: if diversity constraint left batch underfull, pad with greedy top-k
            if len(selected) < self.batch_size:
                for idx in sorted_idx:
                    if idx not in selected:
                        selected.append(idx)
                    if len(selected) >= self.batch_size:
                        break
            top_idx = np.array(selected[:self.batch_size])

            batch_df    = candidates_df.iloc[top_idx].reset_index(drop=True)
            batch_X     = candidates_X[top_idx]
            batch_mu    = mu[top_idx]
            batch_sigma = sigma[top_idx]

            self._X_observed.append(batch_X)
            self._df_observed.append(batch_df)
            self._Y_observed.append(batch_mu)
            self._sigma_observed.append(batch_sigma)

            self._update_pareto()
            hv = self._current_hypervolume()
            self._hv_history.append(hv)

            if self.verbose:
                pf = self._current_pareto_front()
                print(f"  EHVI top score:      {ehvi[top_idx[0]]:.4f}")
                print(f"  Pareto front size:   {len(pf)}")
                print(f"  Hypervolume:         {hv:.4f}  "
                      f"(Δ={hv - self._hv_history[-2]:+.4f})")
                # Print best predicted values per objective
                for j, name in enumerate(self.objective_names):
                    best_val = (self._current_pareto_front()[:, j] * self._signs[j]).max()
                    print(f"  Best {name:30s}: {best_val:.3f}")

        # ── Compile results ───────────────────────────────────────────
        all_X   = np.vstack(self._X_observed)
        all_df  = pd.concat(self._df_observed, ignore_index=True)
        all_Y   = np.vstack(self._Y_observed)     # maximization space
        all_sig = np.vstack(self._sigma_observed)

        # Convert back to original objective space
        all_Y_orig  = all_Y  * self._signs[None, :]

        self._update_pareto()
        pareto_df   = all_df[self._pareto_mask].reset_index(drop=True)
        pareto_Y    = all_Y_orig[self._pareto_mask]
        pareto_sig  = all_sig[self._pareto_mask]

        # Sort Pareto front by first objective descending
        sort_idx    = np.argsort(pareto_Y[:, 0])[::-1]
        pareto_df   = pareto_df.iloc[sort_idx].reset_index(drop=True)
        pareto_Y    = pareto_Y[sort_idx]
        pareto_sig  = pareto_sig[sort_idx]

        if self.verbose:
            self._print_summary(pareto_df, pareto_Y, pareto_sig)

        return MOBOResult(
            pareto_candidates=pareto_df,
            pareto_objectives=pareto_Y,
            pareto_uncertainties=pareto_sig,
            hypervolume_history=self._hv_history,
            all_candidates=all_df,
            all_objectives=all_Y_orig,
            ref_point=self.ref_point * self._signs,
            objective_names=self.objective_names,
        )

    def _print_summary(self, pareto_df, pareto_Y, pareto_sig):
        print(f"\n{'═'*60}")
        print("FINAL PARETO FRONT")
        print(f"{'═'*60}")
        print(f"Total Pareto-optimal designs: {len(pareto_df)}")
        print(f"Final hypervolume: {self._hv_history[-1]:.4f}")
        print(f"Hypervolume improvement: "
              f"{self._hv_history[-1] - self._hv_history[0]:+.4f}")
        print()

        for i in range(min(len(pareto_df), 10)):
            print(f"  Design {i+1:2d}:")
            for col in self.search_space.param_names:
                val = pareto_df.loc[i, col]
                if isinstance(val, float):
                    print(f"    {col:25s}: {val:.3f}")
                else:
                    print(f"    {col:25s}: {val}")
            for j, name in enumerate(self.objective_names):
                print(f"    {name:25s}: {pareto_Y[i, j]:.3f} "
                      f"± {pareto_sig[i, j]:.3f}")
            print()



# ─────────────────────────────────────────────────────────────────────────────
# Standalone Post-Run Diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_mobo(result: MOBOResult) -> None:
    """
    Call on any MOBOResult to diagnose Pareto collapse and
    validate optimization health. Safe to call after run_mobo().

    Usage:
        result = mobo.run()
        diagnose_mobo(result)
    """
    SEP  = "═" * 60
    SEP2 = "─" * 60
    Y    = result.all_objectives          # (N, n_obj) original scale
    pY   = result.pareto_objectives       # (k, n_obj)
    sig  = result.pareto_uncertainties
    hv   = result.hypervolume_history
    names = result.objective_names

    print(f"\n{SEP}")
    print("  MOBO POST-RUN DIAGNOSTIC")
    print(SEP)

    # ── Pareto front summary ──────────────────────────────────────────────────
    print(f"\n  Pareto front size : {len(pY)}")
    print(f"  Total evaluated   : {len(Y)}")
    print(f"  Final hypervolume : {hv[-1]:.6f}")
    print(f"  HV improvement    : {hv[-1] - hv[0]:+.6f}")
    print(f"  Converged after   : {len(hv)} iterations")

    # ── Objective correlation (full evaluated set) ────────────────────────────
    print(f"\n{SEP2}")
    print("  OBJECTIVE CORRELATION  (full evaluated set)")
    print(SEP2)
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            valid = np.isfinite(Y[:, i]) & np.isfinite(Y[:, j])
            corr  = float(np.corrcoef(Y[valid, i], Y[valid, j])[0, 1])
            severity = ("CRITICAL" if abs(corr) > 0.95 else
                        "HIGH"     if abs(corr) > 0.85 else
                        "MODERATE" if abs(corr) > 0.70 else "OK")
            flag = f"  *** {severity} ***" if severity != "OK" else ""
            print(f"    corr({n1[:22]}, {n2[:22]}) = {corr:+.4f}{flag}")
    if len(names) == 1:
        print("    Only one objective — no correlation check")

    # ── Pareto front diversity ────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  PARETO FRONT OBJECTIVE RANGES")
    print(SEP2)
    for i, name in enumerate(names):
        if len(pY) == 0:
            print(f"    {name[:35]}: (empty Pareto front)")
            continue
        lo, hi = pY[:, i].min(), pY[:, i].max()
        spread = hi - lo
        flag   = "  *** NO SPREAD — collapsed ***" if spread < 1e-6 else ""
        print(f"    {name[:35]}: [{lo:.4f}, {hi:.4f}]  spread={spread:.6f}{flag}")

    # ── Uncertainty on Pareto candidates ─────────────────────────────────────
    print(f"\n{SEP2}")
    print("  UNCERTAINTY ON PARETO CANDIDATES")
    print(SEP2)
    for i, name in enumerate(names):
        if len(sig) == 0:
            continue
        s = sig[:, i]
        flag = "  *** NEAR-ZERO — UQ not working ***" if s.mean() < 1e-4 else ""
        print(f"    {name[:35]}: mean={s.mean():.4f}  "
              f"min={s.min():.4f}  max={s.max():.4f}{flag}")

    # ── Hypervolume convergence ───────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  HYPERVOLUME CONVERGENCE")
    print(SEP2)
    if len(hv) >= 2:
        deltas      = np.diff(hv)
        n_improve   = (deltas > 1e-4).sum()
        n_flat      = (deltas <= 1e-4).sum()
        converged   = deltas[-3:].max() < 1e-4 if len(deltas) >= 3 else False
        print(f"    Improving iterations : {n_improve}")
        print(f"    Flat iterations      : {n_flat}")
        print(f"    Converged (last 3)   : {converged}")
        if n_improve == 0:
            print("    *** WARNING: HV never improved — EHVI may be degenerate ***")
        elif n_flat > n_improve * 2:
            print("    *** NOTE: Mostly flat — consider more iterations or "
                  "wider search bounds ***")
    for it, h in enumerate(hv):
        bar_len = int(40 * h / max(hv[-1], 1e-9))
        bar     = "█" * bar_len
        print(f"    [{it:3d}] {h:.4f}  {bar}")

    # ── Reference point check ─────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  REFERENCE POINT CHECK")
    print(SEP2)
    for i, name in enumerate(names):
        ref_i = result.ref_point[i]
        y_min = Y[:, i].min()
        ok    = ref_i < y_min
        flag  = "" if ok else "  *** BAD — increase n_initial or lower ref_point ***"
        print(f"    {name[:35]}: ref={ref_i:.4f}  "
              f"obs_min={y_min:.4f}  {'OK' if ok else 'BAD'}{flag}")

    print(f"\n{SEP}\n")


def pareto_grid_sample(
    result: 'MOBOResult',
    n_grid:  int   = 30,
    epsilon: float = 0.0,
) -> 'MOBOResult':
    """
    Expand a collapsed Pareto front by grid-sampling the tradeoff curve.

    When a strict Pareto front has only 2 points (common when one surrogate
    objective has low discrimination), this function:
      1. Sweeps n_grid evenly-spaced weight vectors across the objective simplex
      2. For each weight w, finds the best candidate in result.all_candidates
         under augmented Chebyshev scalarization
      3. Returns a new MOBOResult whose Pareto set is the union of these
         weight-optimal candidates (deduplicated)

    This is equivalent to computing the full Pareto front approximation by
    weighted-sum / Chebyshev scalarization — standard practice in MOO papers.

    epsilon : fractional tolerance for near-Pareto inclusion.
        0.0 = strict (only Chebyshev-optimal per weight)
        0.05 = include candidates within 5%% of the best score per weight
    """
    import pandas as pd

    Y_all  = result.all_objectives          # (N, n_obj) original scale
    df_all = result.all_candidates          # (N, features)
    sig_all = result.all_uncertainties
    names  = result.objective_names
    n_obj  = len(names)

    # ── Normalize to [0, 1] per objective ─────────────────────────────────────
    y_min = Y_all.min(0)
    y_max = Y_all.max(0)
    y_rng = np.maximum(y_max - y_min, 1e-9)
    Y_norm = (Y_all - y_min[None, :]) / y_rng[None, :]   # (N, n_obj)

    # ── Sweep weight vectors ───────────────────────────────────────────────────
    # For 2 objectives: w = [t, 1-t] for t in linspace(0,1,n_grid)
    # For k objectives: use simplex grid
    if n_obj == 2:
        ts      = np.linspace(0, 1, n_grid)
        weights = np.column_stack([ts, 1 - ts])   # (n_grid, 2)
    else:
        # Uniform simplex grid via Dirichlet
        rng     = np.random.default_rng(42)
        weights = rng.dirichlet(np.ones(n_obj), size=n_grid)

    rho = 0.05   # augmentation coefficient (standard in ParEGO)

    selected_indices = set()
    for w in weights:
        # Augmented Chebyshev (maximisation form):
        #   score = min_j[ w_j * Y_norm_j ] + rho * sum_j[ w_j * Y_norm_j ]
        cheby  = np.min(w[None, :] * Y_norm, axis=1) + rho * (w[None, :] * Y_norm).sum(1)
        best   = cheby.max()
        thresh = best - epsilon * abs(best + 1e-12)
        mask   = cheby >= thresh
        selected_indices.update(np.where(mask)[0].tolist())

    idx = np.array(sorted(selected_indices))

    # Sort by first objective descending (brain delivery)
    sort_order = np.argsort(Y_all[idx, 0])[::-1]
    idx        = idx[sort_order]

    # Build new result preserving all_* arrays
    new_result = MOBOResult(
        pareto_candidates     = df_all.iloc[idx].reset_index(drop=True),
        pareto_objectives     = Y_all[idx],
        pareto_uncertainties  = sig_all[idx],
        all_candidates        = df_all,
        all_objectives        = Y_all,
        all_uncertainties     = sig_all,
        hypervolume_history   = result.hypervolume_history,
        objective_names       = names,
        ref_point             = result.ref_point,
    )
    return new_result

# ─────────────────────────────────────────────────────────────────────────────
# Results Export
# ─────────────────────────────────────────────────────────────────────────────

def export_results(
    result:                MOBOResult,
    prefix:                str  = 'mobo',
    conformal_calibrators: Dict = None,
) -> None:
    """
    Export MOBO results to CSV and plot hypervolume history.

    Generates:
        {prefix}_pareto_formulations.csv  — synthesis-ready Pareto designs
        {prefix}_hypervolume.png          — convergence plot

    Parameters
    ----------
    conformal_calibrators : dict, optional
        {objective_name: ConformalCalibrator} mapping.
        When provided, adds columns conformal_lo_<obj> and conformal_hi_<obj>
        to the output CSV — prediction intervals with guaranteed finite-sample
        marginal coverage, stronger than bootstrap CIs.
        These replace the standard ci_* columns as the primary reported
        uncertainty for each objective.
    """
    # Build output dataframe
    obj_df = pd.DataFrame(
        result.pareto_objectives,
        columns=[f'pred_{n}' for n in result.objective_names]
    )
    ci_df = pd.DataFrame(
        result.pareto_uncertainties,
        columns=[f'ci_{n}' for n in result.objective_names]
    )
    out_df = pd.concat([result.pareto_candidates, obj_df, ci_df], axis=1)

    # ── Conformal prediction intervals (distribution-free coverage guarantee) ──
    if conformal_calibrators:
        for obj_name, cal in conformal_calibrators.items():
            if obj_name not in result.objective_names:
                continue
            j      = result.objective_names.index(obj_name)
            mu_j   = result.pareto_objectives[:, j]
            lo, hi = cal.predict_interval(mu_j)
            out_df[f'conformal_lo_{obj_name}'] = np.round(lo, 6)
            out_df[f'conformal_hi_{obj_name}'] = np.round(hi, 6)
            print(f"  [Conformal] {obj_name}: ±{cal.half_width:.4f}  "
                  f"({cal.coverage:.0%} coverage, n_cal={cal._n_cal})")
    out_df.index.name = 'design_rank'

    # ── Quality flags ─────────────────────────────────────────────────────────
    # Flag 1: clipped predictions (brain delivery = exactly 0.0 after clip)
    # These were negative GPBoost extrapolations — not real predictions.
    brain_col = f'pred_{result.objective_names[-1]}'
    if brain_col in out_df.columns:
        out_df['flag_clipped'] = (out_df[brain_col] == 0.0).astype(int)

    # Flag 2: extrapolation beyond training bounds
    try:
        # Use local fallback bounds — avoids fragile cross-file import
        BRAIN_DELIVERY_TRAIN_BOUNDS = (0.0, 0.693)
        LC_TRAIN_BOUNDS             = (0.0, 36.5)
        lc_col = f'pred_{result.objective_names[0]}'
        out_df['flag_extrapolated'] = 0
        if lc_col in out_df.columns:
            lo, hi = LC_TRAIN_BOUNDS
            out_df['flag_extrapolated'] |= (
                (out_df[lc_col] < lo) | (out_df[lc_col] > hi)
            ).astype(int)
        if brain_col in out_df.columns:
            lo, hi = BRAIN_DELIVERY_TRAIN_BOUNDS
            out_df['flag_extrapolated'] |= (
                (out_df[brain_col] < lo) | (out_df[brain_col] > hi)
            ).astype(int)
    except ImportError:
        pass

    # ── Print flagged summary ──────────────────────────────────────────────────
    n_clipped = out_df.get('flag_clipped', pd.Series(0)).sum()
    n_extrap  = out_df.get('flag_extrapolated', pd.Series(0)).sum()
    if n_clipped > 0:
        print(f"  [QA] WARNING: {n_clipped} candidate(s) have clipped brain delivery "
              f"(negative GPBoost prediction → floored to 0). "
              f"These are extrapolations — treat with caution.")
    if n_extrap > 0:
        print(f"  [QA] WARNING: {n_extrap} candidate(s) predict outside training bounds "
              f"— marked flag_extrapolated=1 in CSV.")
    if n_clipped == 0 and n_extrap == 0:
        print(f"  [QA] All predictions within training bounds.")

    csv_path = f'{prefix}_pareto_formulations.csv'
    out_df.to_csv(csv_path)
    print(f"Pareto formulations saved: {csv_path}")
    print(out_df.to_string())

    # Hypervolume convergence plot
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(14, 5))
        gs = gridspec.GridSpec(1, 3, figure=fig)

        # Hypervolume history
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(result.hypervolume_history, marker='o', color='#2196F3',
                 linewidth=2, markersize=4)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5,
                    label='Init → EHVI')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Hypervolume')
        ax1.set_title('MOBO Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Pareto front scatter (first two objectives)
        ax2 = fig.add_subplot(gs[1])
        obj_names = result.objective_names
        all_Y = result.all_objectives
        pareto_Y = result.pareto_objectives

        ax2.scatter(all_Y[:, 0], all_Y[:, 1],
                    alpha=0.2, s=10, color='gray', label='All evaluated')
        ax2.scatter(pareto_Y[:, 0], pareto_Y[:, 1],
                    alpha=0.9, s=50, color='#E53935', label='Pareto front',
                    zorder=5)
        ax2.errorbar(
            pareto_Y[:, 0], pareto_Y[:, 1],
            xerr=result.pareto_uncertainties[:, 0],
            yerr=result.pareto_uncertainties[:, 1],
            fmt='none', color='#E53935', alpha=0.4, capsize=3
        )
        ax2.set_xlabel(obj_names[0])
        ax2.set_ylabel(obj_names[1] if len(obj_names) > 1 else '')
        ax2.set_title('Pareto Front')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Top designs bar chart
        ax3 = fig.add_subplot(gs[2])
        n_show = min(10, len(pareto_Y))
        x = np.arange(n_show)
        width = 0.35
        vals0 = pareto_Y[:n_show, 0]
        vals1 = pareto_Y[:n_show, 1] if len(obj_names) > 1 else None

        ax3.bar(x - width/2, vals0, width, label=obj_names[0],
                color='#2196F3', alpha=0.8)
        if vals1 is not None:
            ax3.bar(x + width/2, vals1, width,
                    label=obj_names[1], color='#E53935', alpha=0.8)
        ax3.set_xlabel('Design rank')
        ax3.set_ylabel('Predicted value')
        ax3.set_title('Top Pareto Designs')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'D{i+1}' for i in range(n_show)])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = f'{prefix}_hypervolume.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Plot saved: {plot_path}")

    except ImportError:
        print("matplotlib not available — skipping plots")


# ─────────────────────────────────────────────────────────────────────────────
# Cascading MOBO Components  (used by run_mobo_v3)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchSpace:
    """
    Generalised search space: continuous, categorical, binary params plus
    optional lookup tables. fixed_features pins specific columns to a
    constant value during sampling (used to fix drug properties per run).

    lookup_col_map : {table_name: (source_col, output_col)}
        After sampling, each entry derives a numeric output column by mapping
        the source column through the named lookup table.
        e.g. {'surfactant_hlb': ('surfactant_type', 'surfactant_HLB')}
    """
    continuous:          Dict[str, Tuple[float, float]] = field(default_factory=dict)
    categorical:         Dict[str, List]                = field(default_factory=dict)
    binary:              Dict[str, List]                = field(default_factory=dict)
    lookup_tables:       Dict[str, Dict]                = field(default_factory=dict)
    fixed_features:      Dict                           = field(default_factory=dict)
    lookup_col_map:      Dict[str, Tuple[str, str]]     = field(default_factory=dict)
    # receptor → canonical ligand mapping.  When provided, 'ligand' is
    # deterministically derived from the sampled 'receptor' so invalid
    # pairs are impossible by construction (no retry loop needed).
    canonical_ligand_map: Dict[str, str]                = field(default_factory=dict)

    def sample(self, n: int, seed: int = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows: Dict = {}
        for param, (lo, hi) in self.continuous.items():
            rows[param] = (np.full(n, self.fixed_features[param])
                           if param in self.fixed_features
                           else rng.uniform(lo, hi, n))
        for param, options in self.categorical.items():
            rows[param] = ([self.fixed_features[param]] * n
                           if param in self.fixed_features
                           else list(rng.choice(options, n)))
        for param, options in self.binary.items():
            rows[param] = (np.full(n, self.fixed_features[param])
                           if param in self.fixed_features
                           else rng.choice(options, n))
        # ── Enforce biological validity: receptor → canonical ligand ─────────
        # canonical_ligand_map (passed at SearchSpace construction from
        # RECEPTOR_TO_CANONICAL_LIGAND in the run script) deterministically
        # sets 'ligand' from the sampled 'receptor', making invalid pairs
        # impossible by construction — no retry loop, no external import.
        #
        # IMPORTANT: df is built AFTER this block so the DataFrame holds the
        # corrected ligand values.  The old code modified rows[] but built df
        # before the correction, so df always had the unconstrained values —
        # this is the root cause of glucose+integrin appearing in designs.
        if 'receptor' in rows and self.canonical_ligand_map:
            rows['ligand'] = [
                self.canonical_ligand_map.get(str(r), 'none')
                for r in rows['receptor']
            ]

        df = pd.DataFrame(rows)

        # Apply lookup tables to derive numeric columns from categorical keys
        for table_name, (src_col, out_col) in self.lookup_col_map.items():
            if table_name in self.lookup_tables and src_col in df.columns:
                df[out_col] = df[src_col].map(
                    self.lookup_tables[table_name]
                ).fillna(0.0)

        return df

    @property
    def param_names(self) -> List[str]:
        return list(self.continuous) + list(self.categorical) + list(self.binary)


class Surrogate:
    """
    General surrogate for sklearn-compatible regressors (e.g. CatBoost).
    uncertainty_method='bootstrap' fits an ensemble for std estimates;
    uncertainty_method='fixed'     uses uncertainty_fraction * |mu|.
    """

    def __init__(
        self,
        fitted_model,
        name: str,
        uncertainty_method:   str         = 'bootstrap',
        X_train:              np.ndarray  = None,
        y_train:              np.ndarray  = None,
        n_bootstrap:          int         = 30,
        uncertainty_fraction: float       = 0.10,
    ):
        self.model                = fitted_model
        self.name                 = name
        self.uncertainty_method   = uncertainty_method
        self.X_train              = X_train
        self.y_train              = y_train
        self.n_bootstrap          = n_bootstrap
        self.uncertainty_fraction = uncertainty_fraction
        self._bootstrap_models    = None

    def _fit_bootstrap(self):
        from catboost import CatBoostRegressor
        rng  = np.random.default_rng(42)
        iters = self.model.get_param('iterations') or 300
        self._bootstrap_models = []
        for i in range(self.n_bootstrap):
            idx = rng.integers(0, len(self.X_train), len(self.X_train))
            m = CatBoostRegressor(iterations=iters, verbose=0, random_seed=i)
            m.fit(self.X_train[idx], self.y_train[idx])
            self._bootstrap_models.append(m)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.uncertainty_method == 'bootstrap':
            if self._bootstrap_models is None:
                self._fit_bootstrap()
            preds = np.array([m.predict(X) for m in self._bootstrap_models])
            return preds.mean(axis=0), np.maximum(preds.std(axis=0), 1e-6)
        mu    = self.model.predict(X)
        sigma = np.maximum(np.abs(mu) * self.uncertainty_fraction, 1e-6)
        return mu, sigma


class TabPFNSurrogate:
    """Wraps a fitted TabPFNGPModel for use inside CascadingSurrogate."""

    def __init__(
        self,
        fitted_model,
        name:               str = '',
        study_id_default:   str = 'design',
        carrier_id_default: str = 'PLGA',
    ):
        self.model              = fitted_model
        self.name               = name
        self.study_id_default   = study_id_default
        self.carrier_id_default = carrier_id_default

        # ── In-context update buffer (Fix 1) ─────────────────────────────────
        # Accumulated pseudo-observations from previous MOBO iterations.
        # TabPFN is an in-context learner: appending (X, y) pairs to the
        # training context at inference time shifts the posterior toward
        # the explored region — implementing genuine BO rather than random
        # search.  Buffer is capped at _max_context_size to stay within
        # TabPFN's context limit (~1024 rows).
        self._context_X:           Optional[np.ndarray] = None
        self._context_y:           Optional[np.ndarray] = None
        self._context_study_ids:   Optional[np.ndarray] = None
        self._context_carrier_ids: Optional[np.ndarray] = None
        self._max_context_size:    int = 200

    def update_context(
        self,
        X_new:            np.ndarray,
        y_new:            np.ndarray,
        study_ids_new:    np.ndarray = None,
        carrier_ids_new:  np.ndarray = None,
    ) -> None:
        """
        Append new pseudo-observations to the in-context buffer.

        TabPFN is an in-context learner: passing additional (X, y) pairs as
        training context at inference time shifts the posterior for the next
        prediction call.  Using predicted means as pseudo-labels is the
        standard Thompson-sampling regime when no ground-truth labels are
        available yet.

        Buffer is capped at _max_context_size (most-recent rows kept) to
        stay within TabPFN's context limit.
        """
        n    = len(X_new)
        sids = (study_ids_new  if study_ids_new  is not None
                else np.array([self.study_id_default]   * n))
        cids = (carrier_ids_new if carrier_ids_new is not None
                else np.array([self.carrier_id_default] * n))

        if self._context_X is None:
            self._context_X           = X_new.copy()
            self._context_y           = y_new.copy()
            self._context_study_ids   = sids.copy()
            self._context_carrier_ids = cids.copy()
        else:
            self._context_X           = np.vstack([self._context_X, X_new])
            self._context_y           = np.concatenate([self._context_y, y_new])
            self._context_study_ids   = np.concatenate([self._context_study_ids, sids])
            self._context_carrier_ids = np.concatenate([self._context_carrier_ids, cids])

        # Trim to max_context_size — keep most-recent (most informative)
        if len(self._context_X) > self._max_context_size:
            self._context_X           = self._context_X[-self._max_context_size:]
            self._context_y           = self._context_y[-self._max_context_size:]
            self._context_study_ids   = self._context_study_ids[-self._max_context_size:]
            self._context_carrier_ids = self._context_carrier_ids[-self._max_context_size:]

    def predict(
        self,
        X:           np.ndarray,
        study_ids:   np.ndarray = None,
        carrier_ids: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(X)
        if study_ids  is None: study_ids   = np.array([self.study_id_default]   * n)
        if carrier_ids is None: carrier_ids = np.array([self.carrier_id_default] * n)
        try:
            # If context has been accumulated, shift TabPFN's posterior by
            # injecting pseudo-observations as extra training context.
            if self._context_X is not None and len(self._context_X) > 0:
                try:
                    mu, sigma = self.model.predict_with_uncertainty_with_extra_context(
                        X_test            = X,
                        study_ids_test    = study_ids,
                        carrier_ids_test  = carrier_ids,
                        X_extra           = self._context_X,
                        y_extra           = self._context_y,
                        study_ids_extra   = self._context_study_ids,
                        carrier_ids_extra = self._context_carrier_ids,
                    )
                except (AttributeError, NotImplementedError):
                    # Fallback: base prediction (context not injected).
                    # Log once so the user knows context is being silently ignored.
                    if not getattr(self, '_warned_no_extra_context', False):
                        print(
                            "  [TabPFNSurrogate] predict_with_uncertainty_with_extra_context"
                            " not available — using base prediction (context ignored)."
                        )
                        self._warned_no_extra_context = True
                    mu, sigma = self.model.predict_with_uncertainty(X, study_ids, carrier_ids)
            else:
                mu, sigma = self.model.predict_with_uncertainty(X, study_ids, carrier_ids)
            if np.all(np.isnan(sigma)):
                sigma = np.full(n, np.nanstd(mu) * 0.3)
        except Exception as e:
            print(f"  [TabPFNSurrogate] uncertainty failed ({e}), using point predictions")
            mu    = self.model.predict(X, study_ids, carrier_ids)
            sigma = np.full(n, max(float(np.nanstd(mu)) * 0.3, 1e-6))

        # Enforce a minimum sigma = 5% of |mu| so EHVI never collapses to
        # pure exploitation.  30% was over-compensating for the GPBoost
        # unseen-group constant-sigma case — with mu ≈ 0.4 (log1p scale)
        # that floor was ≈ 0.12, comparable to the entire brain-delivery
        # dynamic range [0, 0.693], obliterating the exploration/exploitation
        # distinction and causing the 45x EHVI weight imbalance.
        # 5% is sufficient as a numerical safety margin.
        sigma_floor = np.maximum(np.abs(mu) * 0.05, 1e-4)
        return mu, np.maximum(sigma, sigma_floor)


class CascadingSurrogate:
    """
    Two-stage surrogate with Monte Carlo uncertainty propagation.

    Stage 1: predict LC (and any other stage-1 targets) from formulation features.
    Stage 2: predict brain delivery using formulation features + sampled LC.
    Uncertainty from stage 1 propagates into stage 2 via MC sampling.
    """

    def __init__(
        self,
        stage1_surrogates:   Dict,
        stage2_surrogate:    TabPFNSurrogate,
        cascade_vars:        Dict[str, str],   # {stage1_name: stage2_col_name}
        stage1_feature_cols: List[str],
        stage2_feature_cols: List[str],
        n_cascade_samples:   int = 30,
        column_aliases:      Dict[str, str] = None,  # {search_space_col: stage2_col}
        stage1_encodings:    Dict[str, Dict] = None, # {col: {val: int}} from training
        stage2_encodings:    Dict[str, Dict] = None, # {col: {val: int}} from training
        stage2_imputation:   Dict            = None, # {col: training_mean} for missing cols
        stage1_preprocess:   object          = None, # callable(df) -> df; e.g. add degradation features
    ):
        self.stage1_surrogates   = stage1_surrogates
        self.stage2_surrogate    = stage2_surrogate
        self.cascade_vars        = cascade_vars
        self.stage1_feature_cols = stage1_feature_cols
        self.stage2_feature_cols = stage2_feature_cols
        self.n_cascade_samples   = n_cascade_samples
        self.column_aliases      = column_aliases or {}
        self.stage1_encodings    = stage1_encodings or {}
        self.stage2_encodings    = stage2_encodings or {}
        self.stage2_imputation   = stage2_imputation or {}
        self.stage1_preprocess   = stage1_preprocess

    @staticmethod
    def _extract(df: pd.DataFrame, cols: List[str],
                 encodings: Dict[str, Dict] = None,
                 imputation_defaults: Dict = None) -> np.ndarray:
        """
        Extract cols from df, integer-encode object columns, fill missing
        with training-mean imputation defaults (not zero).

        encodings : pre-built {col: {category: int}} from training data.
            When provided, ensures MOBO candidates receive the same integer
            codes the model was trained on. Unknown categories map to -1.
            Without this, fresh codes are computed per-batch — inconsistent
            with training and silent bug.

        imputation_defaults : {col: value} — training-set mean/mode per column.
            Critical for columns like EE, PDI, zeta that are not search space
            variables. Filling with 0 places candidates severely out-of-
            distribution (e.g. EE=0 vs training range 40-95%), causing
            TabPFN to output near-constant predictions and EHVI to collapse.
        """
        imputation_defaults = imputation_defaults or {}
        out = pd.DataFrame(index=df.index)
        for c in cols:
            if c in df.columns:
                out[c] = df[c]
            elif c in imputation_defaults:
                out[c] = imputation_defaults[c]   # training mean — not 0
            else:
                out[c] = 0.0
        for c in out.select_dtypes(include='object').columns:
            if encodings and c in encodings:
                enc = encodings[c]            # training-aligned codes
            else:
                enc = {v: i for i, v in enumerate(out[c].dropna().unique())}
            mapped = out[c].map(enc)
            n_unknown = mapped.isna().sum()
            if n_unknown > 0:
                pct = 100 * n_unknown / max(len(mapped), 1)
                if pct > 50:
                    valid_vals = list(enc.keys())[:8]
                    import warnings
                    warnings.warn(
                        f"[_extract] Column '{c}': {pct:.0f}% of values are "
                        f"unknown (not in training encodings). These map to -1, "
                        f"causing zero variance in this feature. "
                        f"Valid training values: {valid_vals}. "
                        f"Update MY_CATEGORICAL_PARAMS to use these exact strings.",
                        stacklevel=3
                    )
            out[c] = mapped.fillna(-1)
        return out.fillna(0).values.astype(float)

    def predict(self, candidates_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (mu, sigma) each shape (n_candidates, n_objectives).
        Column order: stage-1 objectives first, then stage-2 objective.
        """
        n  = len(candidates_df)
        if self.stage1_preprocess is not None:
            candidates_df = self.stage1_preprocess(candidates_df)
        X1 = self._extract(candidates_df, self.stage1_feature_cols,
                           encodings=self.stage1_encodings)

        s1_mu:    Dict[str, np.ndarray] = {}
        s1_sigma: Dict[str, np.ndarray] = {}
        for name, sur in self.stage1_surrogates.items():
            s1_mu[name], s1_sigma[name] = sur.predict(X1)

        # ── Monte Carlo cascade uncertainty propagation ──────────────────────────
        # Stage-1 predicts LC with uncertainty (mu_LC, sigma_LC).
        # LC feeds into Stage-2 as feature 'DL'. To propagate LC uncertainty:
        #   1. Draw n_cascade_samples LC values per candidate ~ N(mu_LC, sigma_LC)
        #   2. Build Stage-2 feature matrix for ALL samples (n_samples * n rows)
        #   3. Single TabPFN call — no batch size constraint
        #   4. Aggregate via law of total variance:
        #        E[brain] = mean over samples
        #        Var[brain] = E[Var[brain|LC]] + Var[E[brain|LC]]
        #                   = mean(sigma_tabpfn²) + var(mu_samples)
        # This gives coherent full-pipeline uncertainty that correctly grows
        # when LC is uncertain.
        # ─────────────────────────────────────────────────────────────────────

        rng_mc = np.random.default_rng(42)

        def _build_X2(df_candidates, lc_values):
            df_aug = df_candidates.copy()
            for s1_name, s2_col in self.cascade_vars.items():
                df_aug[s2_col] = lc_values[s1_name]
            if self.column_aliases:
                df_aug = df_aug.rename(columns=self.column_aliases)
            return self._extract(df_aug, self.stage2_feature_cols,
                                 encodings=self.stage2_encodings,
                                 imputation_defaults=self.stage2_imputation)

        K = self.n_cascade_samples   # number of MC samples

        # Sample LC values: shape (K, n) — each row is one MC draw.
        # np.tile repeats the (n,) mean/sigma K times → (K, n) before sampling,
        # so rng_mc.normal returns K independent draws per candidate.
        lc_samples = {
            s1_name: rng_mc.normal(
                loc   = np.tile(s1_mu[s1_name],    (K, 1)),          # (K, n)
                scale = np.maximum(np.tile(s1_sigma[s1_name], (K, 1)), 1e-8),
            )
            for s1_name in s1_mu              # shape (K, n) per stage-1 output
        }

        # Build X2 for all K*n rows in one shot
        # Stack candidates K times, each copy gets a different LC draw
        df_tiled = pd.concat(
            [candidates_df] * K, ignore_index=True
        )
        for s1_name, s2_col in self.cascade_vars.items():
            # lc_samples[s1_name] is (K, n) → flatten to (K*n,) in row-major order
            df_tiled[s2_col] = lc_samples[s1_name].ravel(order='C')

        if self.column_aliases:
            df_tiled = df_tiled.rename(columns=self.column_aliases)

        X2_all = self._extract(df_tiled, self.stage2_feature_cols,
                               encodings=self.stage2_encodings,
                               imputation_defaults=self.stage2_imputation)

        # Single TabPFN call — full K*n batch, no limit
        mu2_flat, sig2_flat = self.stage2_surrogate.predict(X2_all)

        # Reshape to (K, n) and aggregate
        mu2_samples  = mu2_flat.reshape(K, n)    # (K, n)
        sig2_samples = sig2_flat.reshape(K, n)   # (K, n) — TabPFN model uncertainty

        # Law of total variance:
        #   E[Var[y|LC]] = mean of per-sample TabPFN variance
        #   Var[E[y|LC]] = variance of per-sample means (cascade sensitivity)
        mu2_agg  = mu2_samples.mean(axis=0)                              # (n,)
        ev_var   = (sig2_samples ** 2).mean(axis=0)                      # E[Var]
        ve_mean  = mu2_samples.var(axis=0)                               # Var[E]
        sig2_agg = np.maximum(np.sqrt(ev_var + ve_mean), 1e-6)          # (n,)

        # ── Atlas correction REMOVED ──────────────────────────────────────────
        # receptor_bbb_expr and receptor_disease_fc are training features.
        # Post-hoc correction double-counts receptor signal and pushes
        # predictions outside the training manifold. Training feature alone
        # is sufficient.

        # ── Physical lower bound only — upper cap removed ─────────────────────
        # %ID/g >= 0 always (physical constraint: delivery can't be negative).
        # Upper ceiling removed: letting the surrogate extrapolate above the
        # training maximum gives the MOBO freedom to discover genuinely novel
        # high-delivery formulations rather than being forced to cluster at the cap.
        mu2_agg = np.maximum(mu2_agg, 0.0)

                # ── one-time diagnostic ───────────────────────────────────────────────
        if not getattr(self, '_diag_done', False):
            self._diag_done = True
            col_stds = X2_all[:n].std(axis=0)   # first n rows = first MC draw
            print("\n  [CascadeDiag] Stage-2 feature variation (first call):")
            for fname, std in zip(self.stage2_feature_cols, col_stds):
                flag = '  *** ZERO ***' if std < 1e-8 else ''
                print(f"    {fname:<30s}  std={std:.4f}{flag}")
            print(f"  [CascadeDiag] cascade_vars:   {self.cascade_vars}")
            print(f"  [CascadeDiag] column_aliases: {self.column_aliases}")
            print(f"  [CascadeDiag] Brain delivery mu:  "
                  f"min={mu2_agg.min():.4f}  max={mu2_agg.max():.4f}  "
                  f"std={mu2_agg.std():.6f}")
            print(f"  [CascadeDiag] Brain delivery sig: "
                  f"min={sig2_agg.min():.4f}  max={sig2_agg.max():.4f}  "
                  f"std={sig2_agg.std():.6f}\n")
        # ─────────────────────────────────────────────────────────────────────
        s1_keys = list(self.stage1_surrogates.keys())
        mu_all  = np.column_stack([s1_mu[k]    for k in s1_keys] + [mu2_agg])
        sig_all = np.column_stack([s1_sigma[k]  for k in s1_keys] + [sig2_agg])
        return mu_all, sig_all


class CascadingMOBO:
    """
    MOBO loop backed by a CascadingSurrogate and EHVI acquisition.

    objectives : {name: {'direction': 'maximize'|'minimize', 'ref_point': float, ...}}
    The 'space' attribute is exposed so callers can set fixed_features after
    construction (e.g. mobo.space.fixed_features = drug_info['fixed_features']).
    """

    def __init__(
        self,
        cascading_surrogate:   CascadingSurrogate,
        objectives:            Dict,
        search_space:          SearchSpace,
        n_initial:             int  = 50,
        n_iterations:          int  = 20,
        batch_size:            int  = 5,
        n_candidates_per_iter:    int   = 1000,
        n_mc_samples:             int   = 128,
        verbose:                  bool  = True,
        drug_name:                str   = None,
        early_stopping_patience:  int   = 5,
        early_stopping_min_delta: float = 1e-4,
        # ── Fix 1: warm-start ─────────────────────────────────────────────────
        # DataFrame of top training formulations (e.g. top-30 by brain delivery).
        # Rows are mixed into the random init pool so the MOBO starts from a
        # non-trivial Pareto baseline instead of rediscovering obvious optima.
        warmstart_df:        Optional[pd.DataFrame] = None,
        # ── Fix 2: Pareto diversity regularization ────────────────────────────
        # Bonus proportional to each candidate's minimum distance to existing
        # Pareto-front points in normalised continuous synthesis space.
        # Penalises candidates that cluster near already-explored formulations.
        diversity_weight:    float                  = 0.25,
        # ── Fix 3: study-centroid exploration bonus ───────────────────────────
        # (n_studies, n_cont_feats) array of training study centroids in the
        # *same* normalised continuous synthesis space used for diversity.
        # Candidates far from all known study regions get a capped bonus that
        # encourages exploration where the surrogate is least trustworthy
        # (directly tied to the ICC=0.407 between-study variance finding).
        study_centroids:     Optional[np.ndarray]   = None,
        centroid_max_weight: float                  = 0.15,
    ):
        self.surrogate                = cascading_surrogate
        self.objectives               = objectives
        self.space                    = search_space
        self.n_initial                = n_initial
        self.n_iterations             = n_iterations
        self.batch_size               = batch_size
        self.n_candidates_per_iter    = n_candidates_per_iter
        self.n_mc_samples             = n_mc_samples
        self.verbose                  = verbose
        self.drug_name                = drug_name
        self.early_stopping_patience  = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.warmstart_df             = warmstart_df
        self.diversity_weight         = diversity_weight
        self.study_centroids          = study_centroids
        self.centroid_max_weight      = centroid_max_weight

        self.objective_names = list(objectives.keys())
        self.n_obj           = len(self.objective_names)
        self._signs          = np.array([
            1.0 if objectives[k]['direction'] == 'maximize' else -1.0
            for k in self.objective_names
        ])
        self.ref_point = np.array([
            objectives[k]['ref_point'] for k in self.objective_names
        ]) * self._signs

        self._df_observed:    List[pd.DataFrame] = []
        self._Y_observed:     List[np.ndarray]   = []
        self._sigma_observed: List[np.ndarray]   = []
        self._hv_history:     List[float]        = []
        self._pareto_mask:    Optional[np.ndarray] = None
        self._obj_min:        Optional[np.ndarray] = None   # set after Phase-1; guards _current_hv
        self._all_df_cache:   Optional[pd.DataFrame] = None  # invalidated on each append

    # ── internal helpers ───────────────────────────────────────────────

    def _get_all_df(self) -> pd.DataFrame:
        """Return pd.concat of all observed DataFrames, using a cache.

        Rebuilding the full DataFrame from the list is O(total rows) and is
        called multiple times per iteration (_update_pareto, _compute_bonuses,
        results compilation).  Cache it and invalidate whenever a new batch is
        appended so callers always see up-to-date data without the concat cost.
        """
        if self._all_df_cache is None:
            self._all_df_cache = pd.concat(self._df_observed, ignore_index=True)
        return self._all_df_cache

    def _invalidate_df_cache(self) -> None:
        self._all_df_cache = None

    def _post_init_diagnostic(self, Y_init: np.ndarray, sigma_init: np.ndarray):
        """
        Fires after Phase-1 random initialisation.
        Checks the four most common causes of Pareto collapse before
        wasting 20 MOBO iterations on a broken setup.

        Y_init    : (n_initial, n_obj) in maximisation space
        sigma_init: (n_initial, n_obj)
        """
        SEP = "─" * 55
        print(f"\n{SEP}")
        print("  [MOBODiag] POST-INIT DIAGNOSTIC")
        print(SEP)

        # ── Check 1: Objective correlation ───────────────────────────────────
        print("\n  CHECK 1 — Objective correlation")
        Y_orig = Y_init * self._signs[None, :]   # back to original scale
        for i, n1 in enumerate(self.objective_names):
            for j, n2 in enumerate(self.objective_names):
                if j <= i:
                    continue
                corr = float(np.corrcoef(Y_orig[:, i], Y_orig[:, j])[0, 1])
                flag = "  *** HIGH — Pareto front may collapse ***" if abs(corr) > 0.85 else ""
                print(f"    corr({n1[:20]}, {n2[:20]}) = {corr:+.4f}{flag}")

        # ── Check 2: Reference point vs observed range ────────────────────────
        print("\n  CHECK 2 — Reference point vs observed Y range")
        ref_orig = self.ref_point * self._signs   # original scale
        for i, name in enumerate(self.objective_names):
            y_min = Y_orig[:, i].min()
            y_max = Y_orig[:, i].max()
            ref_i = ref_orig[i]
            below = ref_i < y_min
            flag  = "" if below else "  *** BAD — ref_point >= observed min ***"
            print(f"    {name[:30]}: ref={ref_i:.4f}  "
                  f"obs=[{y_min:.4f}, {y_max:.4f}]  {'OK' if below else 'BAD'}{flag}")

        # ── Check 3: Objective variance ───────────────────────────────────────
        print("\n  CHECK 3 — Objective variance across init candidates")
        for i, name in enumerate(self.objective_names):
            std = Y_orig[:, i].std()
            flag = "  *** NEAR-ZERO — surrogate is flat ***" if std < 1e-4 else ""
            print(f"    {name[:30]}: std={std:.6f}{flag}")

        # ── Check 4: Sigma (uncertainty) distribution ─────────────────────────
        print("\n  CHECK 4 — Predicted uncertainty (sigma) distribution")
        for i, name in enumerate(self.objective_names):
            s = sigma_init[:, i]
            flag = "  *** NEAR-ZERO — EHVI will exploit single point ***" if s.mean() < 1e-4 else ""
            print(f"    {name[:30]}: mean={s.mean():.6f}  "
                  f"min={s.min():.6f}  max={s.max():.6f}{flag}")

        # ── Check 5: Pareto front size sanity ─────────────────────────────────
        pf = self._current_pareto_front()
        n_init = len(Y_init)
        pct    = 100 * len(pf) / max(n_init, 1)
        print(f"\n  CHECK 5 — Init Pareto front size: {len(pf)}/{n_init} ({pct:.1f}%)")
        if len(pf) == 1:
            print("    *** SINGLE POINT — most likely cause: objectives perfectly "
                  "correlated OR ref_point above observed minimum ***")
        elif len(pf) == n_init:
            print("    *** ALL POINTS ON PARETO — objectives may be independent "
                  "noise; check surrogate is fitted correctly ***")
        else:
            print("    OK — reasonable Pareto diversity in init sample")

        print(f"\n{SEP}\n")

    def _update_pareto(self):
        Y = np.vstack(self._Y_observed)
        # pareto_mask_2obj: O(n log n) for 2-obj case vs O(n²) is_pareto_efficient
        self._pareto_mask = (pareto_mask_2obj(Y) if self.n_obj == 2
                             else is_pareto_efficient(-Y))

        # ── Pareto archive: monotonically accumulates best-ever front ─────────
        # The current front can shrink when new points dominate old members and
        # can be distorted when normalization bounds expand with new extremes.
        # The archive takes the union of the current front + the existing archive
        # and re-filters for non-dominance — HV is guaranteed non-decreasing and
        # EHVI always scores against the strongest baseline seen so far.
        current_Y  = Y[self._pareto_mask]
        all_df     = self._get_all_df()   # cached concat (Change 4)
        current_df = all_df.iloc[np.where(self._pareto_mask)[0]].reset_index(drop=True)

        if not hasattr(self, '_Y_archive') or len(self._Y_archive) == 0:
            self._Y_archive  = current_Y.copy()
            self._df_archive = current_df.copy()
        else:
            combined_Y   = np.vstack([self._Y_archive, current_Y])
            combined_df  = pd.concat([self._df_archive, current_df], ignore_index=True)
            archive_mask = (pareto_mask_2obj(combined_Y) if self.n_obj == 2
                            else is_pareto_efficient(-combined_Y))
            self._Y_archive  = combined_Y[archive_mask]
            self._df_archive = combined_df.iloc[archive_mask].reset_index(drop=True)

    def _current_pareto_front(self) -> np.ndarray:
        # Use archive if available — guarantees monotone HV across all iterations
        if hasattr(self, '_Y_archive') and len(self._Y_archive) > 0:
            return self._Y_archive
        if not self._Y_observed:
            return np.empty((0, self.n_obj))
        return np.vstack(self._Y_observed)[self._pareto_mask]

    def _current_hv(self) -> float:
        pf = self._current_pareto_front()   # (p, n_obj) in maximization space
        if len(pf) == 0:
            return 0.0
        if self._obj_min is None:
            # Normalization not yet initialized (before Phase 1 completes) —
            # fall back to raw HV so the init print still works.
            return hypervolume_2d(pf, self.ref_point)
        # Normalize Pareto front to [0, 1] per objective using Phase-1 range.
        # This ensures LC and brain-delivery contribute equally to HV
        # regardless of their raw scale difference (36.5 vs 0.693).
        pf_orig = pf * self._signs[None, :]
        pf_norm = (pf_orig - self._obj_min[None, :]) / self._obj_range[None, :]
        return hypervolume_2d(pf_norm, np.zeros(self.n_obj))

    def _augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Broadcast space.fixed_features into every row."""
        df = df.copy()
        for k, v in self.space.fixed_features.items():
            df[k] = v
        return df

    # ── acquisition bonuses ────────────────────────────────────────────

    def _compute_bonuses(
        self,
        cands_df: pd.DataFrame,
        raw_acq:  np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-candidate bonuses for Fix 2 (diversity) and Fix 3 (centroids)
        and add them to raw_acq.  Both bonuses operate in normalised continuous
        synthesis space so they are scale-invariant and surrogate-agnostic.

        Fix 2 — Pareto diversity regularization
            bonus_i = diversity_weight * (d_i / max_d)
            where d_i = min distance from candidate i to any current Pareto-front
            candidate in normalised continuous synthesis space.
            Candidates already close to explored Pareto regions get no extra credit.

        Fix 3 — Study-centroid exploration bonus
            bonus_i = centroid_max_weight * (c_i / max_c) * max(|raw_acq|)
            where c_i = distance to nearest training-study centroid.
            Candidates far from all training clusters get a capped bonus,
            encouraging exploration in under-represented regions where the
            surrogate is least trustworthy (ICC=0.407 between-study finding).

        Parameters
        ----------
        cands_df : DataFrame of candidates (n, *)
        raw_acq  : (n,) base acquisition scores (ehvi or chebyshev)

        Returns
        -------
        (n,) augmented acquisition scores
        """
        cont_cols = [c for c in self.space.continuous if c in cands_df.columns]
        if not cont_cols:
            return raw_acq

        bounds  = np.array([self.space.continuous[c] for c in cont_cols])
        lo, hi  = bounds[:, 0], bounds[:, 1]
        span    = np.maximum(hi - lo, 1e-8)

        def _norm(df):
            arr = df[cont_cols].fillna(0.0).values.astype(float)
            return (arr - lo) / span           # (rows, n_cont)

        cand_norm = _norm(cands_df)            # (n, n_cont)
        acq       = raw_acq.copy().astype(float)

        # ── Fix 2: Pareto diversity ───────────────────────────────────────────
        if self.diversity_weight > 0 and len(self._Y_observed) > 0:
            all_obs_df  = self._get_all_df()
            pareto_rows = all_obs_df.iloc[np.where(self._pareto_mask)[0]]
            pf_norm     = _norm(pareto_rows)   # (p, n_cont)

            if len(pf_norm) > 0:
                # Vectorised min-distance: (n, p, d) → (n, p) → (n,)
                diff     = cand_norm[:, None, :] - pf_norm[None, :, :]
                min_dist = np.linalg.norm(diff, axis=2).min(axis=1)   # (n,)
                max_d    = min_dist.max() + 1e-8
                acq     += self.diversity_weight * (min_dist / max_d)

        # ── Fix 3: study-centroid exploration bonus ───────────────────────────
        if self.study_centroids is not None and len(self.study_centroids) > 0:
            n_cf   = self.study_centroids.shape[1]
            c_norm = cand_norm[:, :n_cf]       # match centroid feature dim

            diff_c     = c_norm[:, None, :] - self.study_centroids[None, :, :]
            min_cdist  = np.linalg.norm(diff_c, axis=2).min(axis=1)  # (n,)
            max_cd     = min_cdist.max() + 1e-8
            rel_dist   = min_cdist / max_cd                           # [0, 1]

            # Cap bonus at centroid_max_weight × scale of current acquisition
            acq_scale  = np.abs(acq).max() + 1e-8
            acq       += self.centroid_max_weight * rel_dist * acq_scale

        return acq

    # ── main loop ─────────────────────────────────────────────────────

    def run(self) -> MOBOResult:
        rng   = np.random.default_rng(42)
        label = self.drug_name or 'MOBO'

        if self.verbose:
            print(f"\n{'═'*55}\n  {label.upper()}\n{'═'*55}")

        # Phase 1: initialisation.
        # Sample n_initial random candidates from the search space.
        # If warmstart_df is provided, prepend the warm-start rows so the MOBO
        # begins from a non-trivial Pareto baseline (Fix 1).
        init_df = self._augment(self.space.sample(self.n_initial, seed=42))

        if self.warmstart_df is not None and len(self.warmstart_df) > 0:
            ws_df   = self._augment(self.warmstart_df.copy())
            # Fill any search-space columns missing from warmstart_df with 0.
            for col in init_df.columns:
                if col not in ws_df.columns:
                    ws_df[col] = 0.0
            ws_df   = ws_df[init_df.columns]   # align column order
            init_df = pd.concat([ws_df, init_df], ignore_index=True)
            if self.verbose:
                print(f"  [WarmStart] Prepended {len(ws_df)} training rows → "
                      f"init pool size = {len(init_df)}")

        mu, sigma = self.surrogate.predict(init_df)
        Y_max     = mu * self._signs[None, :]

        self._df_observed.append(init_df)
        self._Y_observed.append(Y_max)
        self._sigma_observed.append(sigma)
        self._invalidate_df_cache()
        self._update_pareto()
        hv = self._current_hv()
        self._hv_history.append(hv)

        if self.verbose:
            print(f"  [Init] Pareto={len(self._current_pareto_front())} | HV={hv:.4f}")

        # Always run post-init diagnostic — catches collapse causes before
        # wasting full iteration budget on a broken configuration.
        self._post_init_diagnostic(Y_max, sigma)

        # ── Adaptive objective normalization (Fix 3) ──────────────────────────
        # Normalize both objectives to [0, 1] using Phase-1 observed range so
        # that hypervolume contributions are comparable across objectives.
        # Without this, LC (range ≈ 36.5) contributes ~94% of HV and brain
        # delivery (range ≈ 0.693) is effectively invisible to ParEGO/EHVI.
        _Y_init_orig    = Y_max * self._signs[None, :]   # back to original scale
        self._obj_min   = _Y_init_orig.min(axis=0)       # (n_obj,) — frozen after Phase-1
        self._obj_max   = _Y_init_orig.max(axis=0)
        self._obj_range = np.maximum(self._obj_max - self._obj_min, 1e-6)
        if self.verbose:
            print(f"\n  [ObjNorm] Objective ranges after Phase-1 init:")
            for _i, _name in enumerate(self.objective_names):
                print(f"    {_name}: [{self._obj_min[_i]:.4f}, "
                      f"{self._obj_max[_i]:.4f}]  "
                      f"range={self._obj_range[_i]:.4f}")

        # ── Backfill init HV with normalized value ────────────────────────────
        # _current_hv() was called BEFORE _obj_range was set → it returned
        # the raw-scale HV (e.g. 43.3).  Now that normalization is ready,
        # recompute and overwrite so the first delta = iter1_hv - init_hv is
        # computed in the same normalized [0,1] space as every subsequent
        # delta.  Without this, delta at iter 1 = 0.99 - 43.3 = -42.3, which
        # immediately burns early-stopping patience before any real work runs.
        _hv_normalized       = self._current_hv()
        self._hv_history[-1] = _hv_normalized
        if self.verbose:
            print(f"  [ObjNorm] Init HV recomputed (normalized): {_hv_normalized:.4f}")

        # ── Acquisition mode selection ────────────────────────────────────────────
        # EHVI requires well-calibrated, input-dependent sigma.
        # GPBoost returns CONSTANT sigma for unseen groups (all MOBO candidates
        # share study_id='design'), causing EHVI to degenerate to single-
        # objective LC maximization — the brain objective becomes invisible.
        #
        # Diagnosis: sigma_brain (~0.16) >> brain_mu_std (~0.08). EHVI weight on
        # brain improvement ≈ 0.065 vs LC ≈ 2.98 → 45x imbalance → Pareto = 2.
        #
        # Fix: random-weight scalarization (ParEGO, Knowles 2006).
        # Each batch member uses a distinct Dirichlet-sampled weight vector
        # applied to MEAN predictions only (sigma-independent). This explicitly
        # constructs candidates spanning the full Pareto front regardless of
        # sigma calibration. Augmented Chebyshev scalarization is used for
        # better Pareto coverage in non-convex regions.
        # ─────────────────────────────────────────────────────────────────────────
        def _sigma_is_homoskedastic(sigma_arr: np.ndarray, tol: float = 0.01) -> bool:
            """True if sigma is effectively constant across candidates (GPBoost unseen group)."""
            for j in range(sigma_arr.shape[1]):
                cv = sigma_arr[:, j].std() / (sigma_arr[:, j].mean() + 1e-12)
                if cv > tol:
                    return False
            return True

        # Phase 2: EHVI-guided OR ParEGO iterations
        no_improve_count = 0
        for it in range(1, self.n_iterations + 1):
            cands_df  = self._augment(
                self.space.sample(self.n_candidates_per_iter,
                                  seed=int(rng.integers(0, 1_000_000)))
            )
            mu, sigma = self.surrogate.predict(cands_df)
            Y_max     = mu * self._signs[None, :]

            # ── Detect homoskedastic sigma and switch to ParEGO ───────────────
            use_pareto = _sigma_is_homoskedastic(sigma)
            if it == 1 and use_pareto and self.verbose:
                print(f"  [AcqSwitch] Sigma is homoskedastic (GPBoost unseen group). "
                      f"Switching from EHVI → ParEGO random scalarization for Pareto coverage.")

            if use_pareto:
                # ── ParEGO: augmented Chebyshev scalarization ─────────────────
                # Normalize candidates to [0,1] using centrally-maintained bounds
                # (self._obj_min/max/range, updated above).  Both objectives
                # therefore contribute equally to HV regardless of raw scale.
                _Y_max_orig = Y_max * self._signs[None, :]   # → original scale
                Y_norm = (_Y_max_orig - self._obj_min[None, :]) / self._obj_range[None, :]  # (n_cands, n_obj)

                # Pre-compute Fix 2 + Fix 3 bonuses once per iteration (they are
                # weight-vector-independent and can be added to each chebyshev
                # score inside the batch loop without recomputing each time).
                # Use zeros as the base so bonuses are expressed in the same
                # normalised [0,1] scale as chebyshev scores.
                _bonus_base = np.zeros(len(cands_df))
                _bonuses    = self._compute_bonuses(cands_df, _bonus_base)

                # ── Stratified weight schedule (Fix 4) ────────────────────
                # Replace random Dirichlet sampling with a deterministic,
                # evenly-spaced grid that covers the full simplex each batch.
                # For 2 objectives and batch_size=5:
                #   [(1,0), (0.75,0.25), (0.5,0.5), (0.25,0.75), (0,1)]
                # Rotate start index by iteration number so successive
                # batches cover different simplex regions (no repeat coverage).
                # This guarantees every part of the Pareto front is targeted
                # at least once per period = batch_size iterations.
                #
                # For n_obj > 2 the 2-column stack below still produces
                # batch_size linearly-interpolated weight vectors; full
                # Das-Dennis simplex is only needed for n_obj >= 3.
                _n_wv    = self.batch_size
                _alphas  = np.linspace(0.0, 1.0, _n_wv)
                # Rotate by iteration so consecutive batches are offset
                _offset  = (it - 1) % _n_wv
                _alphas  = np.roll(_alphas, _offset)
                if self.n_obj == 2:
                    _weight_vectors = np.column_stack([_alphas, 1.0 - _alphas])
                else:
                    # General case: simplex-projection of evenly-spaced grid
                    _weight_vectors = np.tile(_alphas[:, None], (1, self.n_obj))
                    _weight_vectors = (_weight_vectors
                                       / _weight_vectors.sum(axis=1, keepdims=True))

                top_idx_list = []
                used_candidates = set()
                rho = 0.05  # augmented Chebyshev coefficient
                for _batch_i in range(self.batch_size):
                    w = _weight_vectors[_batch_i]
                    # Augmented Chebyshev: min_j[w_j * Y_norm_j] + ρ * Σ w_j*Y_norm_j
                    chebyshev = (np.min(w[None, :] * Y_norm, axis=1)
                                 + rho * (w[None, :] * Y_norm).sum(1)
                                 + _bonuses)
                    sorted_by_cheby = np.argsort(chebyshev)[::-1]
                    for idx in sorted_by_cheby:
                        if int(idx) not in used_candidates:
                            top_idx_list.append(int(idx))
                            used_candidates.add(int(idx))
                            break
                top_idx = np.array(top_idx_list)
            else:
                # ── EHVI (original path, for input-dependent sigma surrogates) ─
                ehvi = expected_hypervolume_improvement(
                    mu          = Y_max,
                    sigma       = sigma,
                    pareto_front= self._current_pareto_front(),
                    ref_point   = self.ref_point,
                    n_samples   = self.n_mc_samples,
                )
                # Apply Fix 2 (diversity) + Fix 3 (centroids) to EHVI scores too
                ehvi = self._compute_bonuses(cands_df, ehvi)
                sorted_idx = np.argsort(ehvi)[::-1]
                selected   = [sorted_idx[0]]
                pf_now     = self._current_pareto_front()
                if len(pf_now) >= 2:
                    obj_range  = pf_now.max(0) - pf_now.min(0)
                    min_spread = 0.10 * np.linalg.norm(obj_range)
                else:
                    min_spread = 0.0
                for idx in sorted_idx[1:]:
                    if len(selected) >= self.batch_size:
                        break
                    if min_spread > 0:
                        Y_sel = Y_max[selected]
                        dists = np.linalg.norm(Y_max[idx] - Y_sel, axis=1)
                        if dists.min() < min_spread:
                            continue
                    selected.append(idx)
                if len(selected) < self.batch_size:
                    for idx in sorted_idx:
                        if idx not in selected:
                            selected.append(idx)
                        if len(selected) >= self.batch_size:
                            break
                top_idx = np.array(selected[:self.batch_size])[:self.batch_size]

            # ── Surrogate posterior update (in-context BO) ──────────────────
            # Push the selected batch as pseudo-observations into the brain
            # surrogate's context buffer so the next iteration's posterior
            # reflects what the MOBO has already explored.  This is what makes
            # this genuine BO rather than repeated random search.
            # Guard: only CascadingSurrogate has stage2_surrogate + _extract.
            if hasattr(self.surrogate, 'stage2_surrogate'):
                try:
                    _selected_df       = cands_df.iloc[top_idx].reset_index(drop=True)
                    _brain_mu_selected = mu[top_idx, -1]   # last obj = brain delivery
                    _X2_selected       = self.surrogate._extract(
                        _selected_df,
                        self.surrogate.stage2_feature_cols,
                        encodings           = self.surrogate.stage2_encodings,
                        imputation_defaults = self.surrogate.stage2_imputation,
                    )
                    self.surrogate.stage2_surrogate.update_context(
                        X_new  = _X2_selected,
                        y_new  = _brain_mu_selected,
                    )
                except Exception as _ctx_err:
                    if self.verbose:
                        print(f"  [InContextUpdate] Warning: {_ctx_err}")

            self._df_observed.append(cands_df.iloc[top_idx].reset_index(drop=True))
            self._Y_observed.append(Y_max[top_idx])
            self._sigma_observed.append(sigma[top_idx])
            self._invalidate_df_cache()
            self._update_pareto()
            hv = self._current_hv()
            self._hv_history.append(hv)

            # Track best Pareto front ever seen (by HV).
            # The strict Pareto front can shrink when new points dominate old ones —
            # correct behavior, but means final front may be smaller than peak.
            # Snapshot best-ever as integer row indices (not a boolean mask) so
            # the indices remain valid after more rows are appended to _Y_observed.
            if not hasattr(self, '_best_hv_seen'):
                self._best_hv_seen      = hv
                self._best_pareto_idx   = np.where(self._pareto_mask)[0].copy()
            elif hv > self._best_hv_seen:
                self._best_hv_seen      = hv
                self._best_pareto_idx   = np.where(self._pareto_mask)[0].copy()

            delta = hv - self._hv_history[-2]
            if self.verbose:
                pf = self._current_pareto_front()
                print(f"  [Iter {it:2d}/{self.n_iterations}] "
                      f"Pareto={len(pf)} | HV={hv:.4f} "
                      f"(Δ={delta:+.4f})")

            # Early stopping
            if delta < self.early_stopping_min_delta:
                no_improve_count += 1
            else:
                no_improve_count = 0
            if no_improve_count >= self.early_stopping_patience:
                if self.verbose:
                    print(f"  [Early stop] No HV improvement ≥ {self.early_stopping_min_delta:.1e} "
                          f"for {self.early_stopping_patience} consecutive iterations.")
                break

        # Compile results
        all_df     = self._get_all_df()
        all_Y      = np.vstack(self._Y_observed)
        all_sig    = np.vstack(self._sigma_observed)
        all_Y_orig = all_Y * self._signs[None, :]

        # Use best-ever Pareto (by HV) rather than final mask.
        # Final Pareto can be smaller than peak if late iterations found
        # points that dominated earlier Pareto members.
        # _best_pareto_idx stores integer row indices — safe to use on the
        # final all_df regardless of how many rows were added after snapshot.
        self._update_pareto()
        # ── Archive-based final results ───────────────────────────────────────
        # The archive is the union of every Pareto front seen during the run,
        # re-filtered for non-dominance.  It can only grow — using it here
        # guarantees we return the best-ever front, not the (potentially
        # shrunken) final-iteration front.
        if hasattr(self, '_df_archive') and len(self._df_archive) > 0:
            pareto_df  = self._df_archive.reset_index(drop=True)
            pareto_Y   = self._Y_archive * self._signs[None, :]
            # Sigma: use per-objective mean of all observed sigmas as a
            # conservative fallback (individual archive rows don't have a
            # 1-to-1 sigma stored because some came from earlier iterations).
            pareto_sig = np.full(
                (len(pareto_df), self.n_obj),
                np.vstack(self._sigma_observed).mean(axis=0),
            )
        else:
            final_idx  = getattr(self, '_best_pareto_idx', np.where(self._pareto_mask)[0])
            pareto_df  = all_df.iloc[final_idx].reset_index(drop=True)
            pareto_Y   = all_Y_orig[final_idx]
            pareto_sig = all_sig[final_idx]

        if self.verbose:
            n_archive = len(self._Y_archive) if hasattr(self, '_Y_archive') else 0
            n_final   = self._pareto_mask.sum()
            best_hv   = getattr(self, '_best_hv_seen', self._hv_history[-1])
            if n_archive != n_final:
                print(f"  [Archive] Returning Pareto archive ({n_archive} pts, "
                      f"HV={best_hv:.4f}) vs final front ({n_final} pts, "
                      f"HV={self._hv_history[-1]:.4f})")

        sort_idx = np.argsort(pareto_Y[:, 0])[::-1]
        return MOBOResult(
            pareto_candidates    = pareto_df.iloc[sort_idx].reset_index(drop=True),
            pareto_objectives    = pareto_Y[sort_idx],
            pareto_uncertainties = pareto_sig[sort_idx],
            hypervolume_history  = self._hv_history,
            all_candidates       = all_df,
            all_objectives       = all_Y_orig,
            all_uncertainties    = all_sig,
            ref_point            = self.ref_point * self._signs,
            objective_names      = self.objective_names,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Running synthetic smoke test with mock surrogates...")

    class MockSurrogate:
        """Mock surrogate for smoke testing."""
        def __init__(self, fn, noise=0.5):
            self.fn = fn
            self.noise = noise
        def predict(self, X):
            mu = self.fn(X)
            return mu, np.full(len(X), self.noise)

    space = PLGASearchSpace()

    # Mock objectives: brain delivery favors small particles + high PEG
    # EE favors high drug loading + specific solvents
    brain_surrogate = MockSurrogate(
        fn=lambda X: (
            -0.01 * X[:, 4]           # particle size (smaller = better)
            + 0.05 * X[:, 5]          # PEG fraction
            + 0.03 * X[:, 1]          # LA:GA ratio
            + np.random.normal(0, 0.3, len(X))
        ),
        noise=0.4,
    )
    ee_surrogate = MockSurrogate(
        fn=lambda X: (
            0.04 * X[:, 2]            # drug loading
            + 0.02 * X[:, 0]          # PLGA MW
            + np.random.normal(0, 0.5, len(X))
        ),
        noise=0.5,
    )

    loop = PLGAMOBOLoop(
        surrogates={
            'brain_delivery_%ID_g':     brain_surrogate,
            'encapsulation_efficiency': ee_surrogate,
        },
        objectives={
            'brain_delivery_%ID_g':     'maximize',
            'encapsulation_efficiency': 'maximize',
        },
        search_space=space,
        ref_point=np.array([-5.0, -5.0]),
        n_initial=30,
        n_iterations=10,
        batch_size=5,
        n_candidates_per_iter=500,
        n_mc_samples=128,
        verbose=True,
    )

    result = loop.run()
    export_results(result, prefix='smoke_test')
