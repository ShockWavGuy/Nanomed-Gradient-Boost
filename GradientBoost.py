import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from LMEM_clean import lmem_clean_target
from scipy.stats import spearmanr
import gpboost as gpb
import optuna
import inspect



def prep_dataset(
    df: pd.DataFrame,
    target_col: str = "logAUC",
    group_col = None,
    one_hot: bool = False,
    feat = ["Weight","logP","Solubility","Pgp","DrugCarrierRatio","Position","Size","Zeta",
        "Route","TPSA","pKa","CarrierNature","Comp1","Comp2"],
    cat = ["Route", "CarrierNature","Comp1","Comp2"]

):

    # Define features
    feature_cols = feat
    categorical_cols = cat

    data = df.copy()

    # Ensure categoricals are string
    for col in categorical_cols:
        data[col] = data[col].astype("string")

    # Optional one-hot for e.g. LightGBM
    if one_hot:
        data = pd.get_dummies(
            data,
            columns=categorical_cols,
            drop_first=True,
        )

    # Columns to keep (target + group + features / dummies)
    if group_col: base_keep = [target_col,group_col]
    else: base_keep = [target_col]
    keep_cols = [c for c in base_keep if c in data.columns]

    # If one_hot=True, dummies expanded; just keep everything except non-feature junk
    if one_hot:
        # keep target, group, and any column that started as a feature or dummy of it
        feature_prefixes = [c for c in feature_cols if c in data.columns]
        keep_cols += [c for c in data.columns if c not in keep_cols and any(
            c == f or c.startswith(f + "_") for f in categorical_cols + [col for col in feature_cols if col not in categorical_cols]
        )]
    else:
        keep_cols += [c for c in feature_cols if c in data.columns]

    return data[keep_cols]

    
def get_cv(cv_type, n_splits=5, groups=None, random_state=42):
    """
    Returns a splitter function for KFold or GroupKFold.
    """
    if cv_type == "group":
        if groups is None:
            raise ValueError("groups must be provided for GroupKFold.")
        # return a function that behaves like KFold.split(X, y, groups)
        def splitter(X, y, groups=groups):
            return GroupKFold(n_splits=n_splits).split(X, y, groups)
        return splitter

    elif cv_type == "kfold":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        def splitter(X, y, groups=None):
            return kf.split(X, y)
        return splitter

    else:
        raise ValueError("cv_type must be 'group' or 'kfold'.")


def get_model(model_type):
    """
    Returns an LGBMRegressor or CatBoostRegressor with reasonable defaults.
    """
    if model_type == "lgbm":
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )

    elif model_type == "catboost":
        return CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_state=42,
            verbose=False
        )

    else:
        raise ValueError("model_type must be 'lgbm' or 'catboost'.")


def cv_regression_lgbm(
    df,
    target_col,
    model_type="lgbm",
    cv_type="group",
    group_col=None,
    binary=False,
    n_splits=5,
    feature_cols=None,
    n_est=500,learn=0.05,max_d=-1,sub=0.9,col=0.9,rand=42
):
    # 1. Determine feature columns
    if feature_cols is None:
        exclude = {target_col}
        if group_col is not None:
            exclude.add(group_col)
        feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y = df[target_col].values

    # 2. Choose CV splitter
    if cv_type == "group":
        if group_col is None:
            raise ValueError("group_col must be provided for group CV.")
        groups = df[group_col].values
        gkf = GroupKFold(n_splits=n_splits)
        splits = gkf.split(X, y, groups)
    elif cv_type == "kfold":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = kf.split(X, y)
    else:
        raise ValueError("cv_type must be 'group' or 'kfold'.")

    # 3. CV loop
    oof_preds = np.zeros(len(df))
    models = []
    r2_scores, rmses, maes, rhos, pvals, AUC = [], [], [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LGBMRegressor(
            n_estimators=n_est,
            learning_rate=learn,
            max_depth=max_d,
            subsample=sub,
            colsample_bytree=col,
            random_state=rand,
            verbose=-1,
            stopping_rounds=50
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        oof_preds[val_idx] = y_pred

        r2 = r2_score(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)
        rho, p_value = spearmanr(y_val, y_pred)
        if binary: AUROC = roc_auc_score(y_val, y_pred)

        r2_scores.append(r2)
        rmses.append(rmse)
        maes.append(mae)
        rhos.append(rho)
        pvals.append(p_value)
        if binary: 
            AUC.append(AUROC)
            print(f"Fold {fold}: R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, Rho={rho:.3f}, P-value={p_value:.3e}, AUROC={AUROC:.3f}")
        else:
            print(f"Fold {fold}: R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, Rho={rho:.3f}, P-value={p_value:.3e}")
        models.append(model)

    print("\nOverall:")
    print(f"R2 mean={np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
    print(f"RMSE mean={np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
    print(f"MAE mean={np.mean(maes):.3f} ± {np.std(maes):.3f}")
    print(f"Rho  mean={np.mean(rhos):.3f} ± {np.std(rhos):.3f}")
    print(f"AUROC mean={np.mean(AUC):.3f} ± {np.std(AUC):.3f}")

    return {
        "models": models,
        "oof_preds": oof_preds,
        "r2_scores": r2_scores,
        "rmses": rmses,
        "maes": maes,
        "rhos": rhos,
        "AUROC": AUC,
        "feature_cols": feature_cols,
    }

def cv_regression_catboost(
    df,
    target_col,
    cat_cols,
    cv_type="group",
    group_col=None,
    binary=False,
    n_splits=5,
    feature_cols=None,
    iterat=75, learn=0.03, dep=6, rand=42,
    leaf_reg=3, rand_str=2, bag_temp=1.0, min_d=10, boot="Bayesian",
    verbose=True,
    # LMEM-specific args
    use_lmem_cleaning=True,
    lmem_fixed_effects=None,
    lmem_reml=True,
    lmem_verbose=False,
    #Loss
    loss_function='RMSE'
):

    if cv_type == "group" and group_col is None:
        raise ValueError("group_col (e.g. 'Study') is required for group CV.")

    # 1. Determine feature columns
    if feature_cols is None:
        exclude = {target_col}
        if group_col is not None:
            exclude.add(group_col)
        feature_cols = [c for c in df.columns if c not in exclude]

    # Default LMEM fixed effects: all feature_cols (you can pass a custom subset)
    if use_lmem_cleaning:
        if group_col is None:
            raise ValueError("group_col is required when use_lmem_cleaning=True.")
        if lmem_fixed_effects is None:
            lmem_fixed_effects = feature_cols.copy()

    # 2. Build X / y
    X = df[feature_cols].copy()
    y = df[target_col].values

    # Ensure categoricals are string dtype (for CatBoost)
    for c in cat_cols:
        if c not in X.columns:
            raise ValueError(f"Categorical column '{c}' not in feature_cols.")
        X[c] = X[c].astype(str)

    cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols]

    # 3. Choose CV splitter
    if cv_type == "group":
        groups = df[group_col].values
        gkf = GroupKFold(n_splits=n_splits)
        splits = gkf.split(X, y, groups)
    elif cv_type == "kfold":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = kf.split(X, y)
    else:
        raise ValueError("cv_type must be 'group' or 'kfold'.")

    # 4. CV loop
    oof_preds = np.zeros(len(df))
    models = []
    r2_scores, rmses, maes, rhos, pvals, AUC = [], [], [], [], [], []

    if use_lmem_cleaning:
        lr2_scores, lrmses, lmaes = [], [], []

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\n[CatBoost CV] Fold {fold}")

        # Fold-specific frames from original df (so LMEM sees everything it needs)
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        # 4a. Targets: either cleaned via LMEM or raw
        if use_lmem_cleaning:
            y_train_clean, y_val_clean, lmem_res = lmem_clean_target(
                train_df=train_df,
                test_df=val_df,
                target=target_col,
                fixed_effects=lmem_fixed_effects,
                group_col=group_col,
                reml=lmem_reml,
                verbose=lmem_verbose,
            )
            y_train_fold = y_train_clean.values
            y_val_fold = y_val_clean.values
        else:
            # Raw (or already transformed) target
            y_train_fold = train_df[target_col].values
            y_val_fold = val_df[target_col].values

        # 4b. Features for CatBoost from pre-built X
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        model = CatBoostRegressor(
            iterations=iterat,
            learning_rate=learn,
            depth=dep,
            loss_function=loss_function,
            random_state=rand,
            verbose=False,
            l2_leaf_reg=leaf_reg,
            random_strength=rand_str,
            bootstrap_type=boot,
            bagging_temperature=bag_temp,
            min_data_in_leaf=min_d,
            early_stopping_rounds=50
        )

        model.fit(
            X_train,
            y_train_fold,
            cat_features=cat_feature_indices,
        )

        print("trained!")
        x_pred = model.predict(X_train)
        print(f"diddy: r2={r2_score(y_train_fold,x_pred)}")

        y_pred = model.predict(X_val)
        oof_preds[val_idx] = y_pred
        print("predicted!")

        # 4c. Evaluate on the same target space we trained on
        r2 = r2_score(y_val_fold, y_pred)
        mse = mean_squared_error(y_val_fold, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_fold, y_pred)
        rho, p_value = spearmanr(y_val_fold, y_pred)
        if binary: AUROC = roc_auc_score(y_val_fold, y_pred)


        r2_scores.append(r2)
        rmses.append(rmse)
        maes.append(mae)
        rhos.append(rho)
        pvals.append(p_value)
        if binary: AUC.append(AUROC)
        if verbose:
            if binary: print(f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, Rho={rho:.3f}, pval={p_value:.3e}, AUROC={AUROC:.3f}")
            else: print(f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, Rho={rho:.3f}, pval={p_value:.3e}")
        models.append(model)

    if verbose:
        print("\n[CatBoost CV] Overall:")
        print(f"R2 mean={np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
        print(f"RMSE mean={np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
        print(f"MAE  mean={np.mean(maes):.3f} ± {np.std(maes):.3f}")
        print(f"Rho  mean={np.mean(rhos):.3f} ± {np.std(rhos):.3f}")
        print(f"AUROC  mean={np.mean(AUC):.3f} ± {np.std(AUC):.3f}")

    return {
        "models": models,
        "oof_preds": oof_preds,
        "r2_scores": r2_scores,
        "rmses": rmses,
        "maes": maes,
        "rhos": rhos,
        "pvals": pvals,
        "AUROC": AUC,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "lmem_fixed_effects": lmem_fixed_effects if use_lmem_cleaning else None,
        "used_lmem_cleaning": use_lmem_cleaning,
    }

def cv_gpboost_native_compat(
    df,
    target_col,
    cat_cols,
    cv_type="group",
    group_col=None,
    binary=False,
    n_splits=5,
    feature_cols=None,
    # training params
    num_boost_round=4000,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=None,
    min_data_in_leaf=10,
    lambda_l2=3.0,
    feature_fraction=1.0,
    bagging_fraction=1.0,
    bagging_freq=0,
    seed=42,
    early_stopping_rounds=50,
    approx="vecchia",
    pred_resp=True,
    # LMEM cleaning
    use_lmem_cleaning=True,
    lmem_fixed_effects=None,
    lmem_reml=True,
    lmem_verbose=False,
    # likelihood
    likelihood=None,
    verbose=True,
):
    if group_col is None:
        raise ValueError("group_col is required for GPBoost random effects (e.g., 'Study').")

    # 1) Features
    if feature_cols is None:
        exclude = {target_col, group_col}
        feature_cols = [c for c in df.columns if c not in exclude]

    if use_lmem_cleaning and lmem_fixed_effects is None:
        lmem_fixed_effects = feature_cols.copy()

    X = df[feature_cols].copy()
    y = df[target_col].values

    for c in cat_cols:
        if c not in X.columns:
            raise ValueError(f"Categorical column '{c}' not in feature_cols.")
        X[c] = X[c].astype("category")

    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    # 2) CV splitter
    if cv_type == "group":
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, y, df[group_col].values)
    elif cv_type == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(X, y)
    else:
        raise ValueError("cv_type must be 'group' or 'kfold'.")

    # 3) Likelihood / objective / metric
    if likelihood is None:
        likelihood = "bernoulli_logit" if binary else "gaussian"

    objective = "binary" if binary else "regression_l2"
    metric = "auc" if binary else "l2"

    if num_leaves is None:
        num_leaves = int(min(2 ** max_depth, 256))

    params = {
        "objective": objective,
        "metric": metric,
        "learning_rate": learning_rate,
        "max_depth": int(max_depth),
        "num_leaves": int(num_leaves),
        "min_data_in_leaf": int(min_data_in_leaf),
        "lambda_l2": float(lambda_l2),
        "feature_fraction": float(feature_fraction),
        "bagging_fraction": float(bagging_fraction),
        "bagging_freq": int(bagging_freq),
        "seed": int(seed),
        "verbose": -1,
    }

    
    oof_preds = np.zeros(len(df))
    boosters, gp_models = [], []

    r2s, rmses, maes, rhos, pvals, aucs = [], [], [], [], [], []

    for fold, (tr, va) in enumerate(splits, 1):
        if verbose:
            print(f"\n[GPBoost CV] Fold {fold}")

        tr_df, va_df = df.iloc[tr].copy(), df.iloc[va].copy()

        # 4a) target cleaning
        if use_lmem_cleaning:
            y_tr, y_va, _ = lmem_clean_target(
                train_df=tr_df,
                test_df=va_df,
                target=target_col,
                fixed_effects=lmem_fixed_effects,
                group_col=group_col,
                reml=lmem_reml,
                verbose=lmem_verbose,
            )
            y_tr, y_va = y_tr.values, y_va.values
        else:
            y_tr, y_va = tr_df[target_col].values, va_df[target_col].values

        X_tr, X_va = X.iloc[tr], X.iloc[va]
        g_tr, g_va = tr_df[group_col].values, va_df[group_col].values

        dtrain = gpb.Dataset(X_tr, label=y_tr, categorical_feature=cat_idx)
        dvalid = gpb.Dataset(X_va, label=y_va, categorical_feature=cat_idx, reference=dtrain)
        use_gp_val = True
        

        # 4c) gp_model with TRAIN group_data (random intercept per group)
        gp_model = gpb.GPModel(group_data=g_tr, likelihood=likelihood, gp_approx=approx)
        gp_model.set_prediction_data(group_data_pred=g_va)
        
        # 4d) train
        booster = gpb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=int(num_boost_round),
            gp_model=gp_model,
            valid_sets=[dvalid],
            valid_names=["valid"],
            use_gp_model_for_validation=use_gp_val,
            early_stopping_rounds=int(early_stopping_rounds),
            verbose_eval=False,
        )

        # 4e) predict with FULL model (trees + RE) on validation
        pred = booster.predict(
            X_va,
            group_data_pred=g_va,
            pred_response=pred_resp,
        )
        y_pred = pred["response_mean"] if isinstance(pred, dict) else pred

        oof_preds[va] = y_pred

        # 4f) metrics on the same target space
        rho, p = spearmanr(y_va, y_pred)
        rhos.append(rho)
        pvals.append(p)

        if binary:
            auc = roc_auc_score(y_va, y_pred)
            aucs.append(auc)
            if verbose:
                print(f"AUROC={auc:.3f}, Rho={rho:.3f}, pval={p:.3e}")
        else:
            r2 = r2_score(y_va, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_va, y_pred)))
            mae = mean_absolute_error(y_va, y_pred)

            r2s.append(r2)
            rmses.append(rmse)
            maes.append(mae)

            if verbose:
                print(f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, Rho={rho:.3f}, pval={p:.3e}")

        boosters.append(booster)
        gp_models.append(gp_model)

    if verbose:
        print("\n[GPBoost CV] Overall:")
        if binary:
            print(f"AUROC mean={np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
            print(f"Rho   mean={np.mean(rhos):.3f} ± {np.std(rhos):.3f}")
        else:
            print(f"R2   mean={np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
            print(f"RMSE mean={np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
            print(f"MAE  mean={np.mean(maes):.3f} ± {np.std(maes):.3f}")
            print(f"Rho  mean={np.mean(rhos):.3f} ± {np.std(rhos):.3f}")

    return {
        "boosters": boosters,
        "gp_models": gp_models,
        "oof_preds": oof_preds,
        "residuals": df[target_col] - oof_preds,
        "r2_scores": r2s,
        "rmses": rmses,
        "maes": maes,
        "rhos": rhos,
        "pvals": pvals,
        "AUROC": aucs,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "likelihood": likelihood,
        "params": params,
    }

def full_regression_catboost(
    df,
    target_col,
    cat_cols,
    group_col=None,
    feature_cols=None,
    iterat=75, learn=0.03, dep=6, rand=42,
    leaf_reg=3, rand_str=2, bag_temp=1.0, min_d=10, boot="Bayesian",
):

    # 1. Determine feature columns
    if feature_cols is None:
        exclude = {target_col}
        if group_col is not None:
            exclude.add(group_col)
        feature_cols = [c for c in df.columns if c not in exclude]

    # Default LMEM fixed effects: all feature_cols

    # 2. Build X / y
    X = df[feature_cols].copy()
    y_raw = df[target_col].values

    # Ensure categoricals are string dtype (for CatBoost)
    for c in cat_cols:
        if c not in X.columns:
            raise ValueError(f"Categorical column '{c}' not in feature_cols.")
        X[c] = X[c].astype("string")

    cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols]

    y_train = y_raw

    # 4. Train CatBoost on full data
    model = CatBoostRegressor(
        iterations=iterat,
        learning_rate=learn,
        depth=dep,
        loss_function="RMSE",
        random_state=rand,
        verbose=False,
        l2_leaf_reg=leaf_reg,
        random_strength=rand_str,
        bootstrap_type=boot,
        bagging_temperature=bag_temp,
        min_data_in_leaf=min_d,
    )

    model.fit(
        X,
        y_train,
        cat_features=cat_feature_indices,
    )

    # 5. In-sample predictions + metrics
    y_pred = model.predict(X)

    r2 = r2_score(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, y_pred)

    print(f"[CatBoost Full-Data] R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

    return {
        "model": model,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "train_metrics": {"r2": r2, "rmse": rmse, "mae": mae},
        "y_train_true": y_train,
        "y_train_pred": y_pred,
    }




def tune_catboost_with_optuna(cat_df, target_col, cat_cols, feat_cols, cv_type="kfold", group_col=None,
                              n_trials=50, n_splits=5, binary=False):
    """
    Run Optuna to tune cv_regression_catboost.
    All dependencies are passed explicitly to avoid NameErrors.
    """
    def objective(trial):
        # --- search space ---
        iterat    = trial.suggest_int("iterat", 300, 6500)
        learn     = trial.suggest_float("learn", 0.01, 0.2, log=True)
        dep       = trial.suggest_int("dep", 2, 10)
        leaf_reg  = trial.suggest_float("leaf_reg", 1.0, 30.0, log=True)
        rand_str  = trial.suggest_float("rand_str", 0.0, 2.0)
        bag_temp  = trial.suggest_float("bag_temp", 0.2, 2.0)
        min_d     = trial.suggest_int("min_d", 5, 60)
        boot      = "Bayesian"

        results_cat = cv_regression_catboost(
            df=cat_df,
            target_col=target_col,
            cat_cols=cat_cols,
            feature_cols=feat_cols,
            use_lmem_cleaning=False,
            cv_type=cv_type,
            group_col=group_col,
            binary=binary,
            n_splits=n_splits,
            verbose=False,
            iterat=iterat,
            learn=learn,
            dep=dep,
            rand=42,
            leaf_reg=leaf_reg,
            rand_str=rand_str,
            bag_temp=bag_temp,
            min_d=min_d,
            boot=boot,
        )
        if binary: 
            AUROC = float(np.mean(results_cat["AUROC"]))
            return AUROC
        else:
            r2_mean = float(np.mean(results_cat["r2_scores"]))
            return r2_mean  

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    if binary: print("Best AUROC:", study.best_value)
    else: print("Best r2:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study


def tune_lgbm_with_optuna(
    lgbm_df,
    target_col,
    n_trials=50,
    cv_type="kfold",
    group_col=None,
    n_splits=5,
):
    """
    Run Optuna to tune cv_regression_lgbm.
    All data-related objects are passed explicitly to avoid scope issues.
    """

    def objective(trial):
        # --- search space ---
        n_est = trial.suggest_int("n_est", 300, 4000)                         # trees
        learn = trial.suggest_float("learn", 0.01, 0.2, log=True)             # learning_rate
        max_d = trial.suggest_categorical("max_d", [-1, 3, 5, 7, 9, 12])      # max_depth
        sub   = trial.suggest_float("sub", 0.5, 1.0)                           # subsample
        col   = trial.suggest_float("col", 0.3, 1.0)                           # colsample_bytree

        rand = 42  # fixed seed for reproducibility

        results_lgbm = cv_regression_lgbm(
            df=lgbm_df,
            target_col=target_col,
            cv_type=cv_type,
            group_col=group_col,
            n_splits=n_splits,
            n_est=n_est,
            learn=learn,
            max_d=max_d,
            sub=sub,
            col=col,
            rand=rand,
        )

        r2_mean = float(np.mean(results_lgbm["r2_scores"]))
        # Optuna will MAXIMIZE this
        return r2_mean

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best r2:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study

import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

def ensemble(
    catdf,
    lgbmdf,
    target_col,
    cat_cols,
    group_col=None,
    binary=False,
    n_splits=5,
    # CatBoost
    iterat=75, learn1=0.03, dep=6, rand1=42,
    leaf_reg=3, rand_str=2, bag_temp=1.0, min_d=10, boot="Bayesian", loss_function='RMSE',
    # LGBM
    n_est=500, learn2=0.05, max_d=-1, sub=0.9, col=0.9, rand2=42,
    # Verbosity
    verbose=True
):
    # 1) Feature columns
    exclude = {target_col}
    if group_col is not None:
        exclude.add(group_col)

    feature_cols_cat  = [c for c in catdf.columns  if c not in exclude]
    feature_cols_lgbm = [c for c in lgbmdf.columns if c not in exclude]

    # 2) Build X / y
    X_c = catdf[feature_cols_cat].copy()
    X_l = lgbmdf[feature_cols_lgbm].copy()
    y   = catdf[target_col].values

    # Ensure categoricals are string dtype (for CatBoost)
    for c in cat_cols:
        if c not in X_c.columns:
            raise ValueError(f"Categorical column '{c}' not in CatBoost feature columns.")
        X_c[c] = X_c[c].astype(str)

    cat_feature_indices = [X_c.columns.get_loc(c) for c in cat_cols]

    # 3) Choose splitter
    if group_col is not None:
        groups = catdf[group_col].values
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X_c, y, groups=groups)
        if verbose:
            n_groups = len(np.unique(groups))
            print(f"[CV] Using GroupKFold with group_col='{group_col}' (unique groups={n_groups})")
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(X_c, y)
        if verbose:
            print("[CV] Using KFold")

    # 4) CV loop
    oof_preds = np.zeros(len(catdf))
    models = []
    r2_scores, rmses, maes, rhos, pvals, AUC = [], [], [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(split_iter, 1):
        if verbose:
            print(f"\n[Ensemble CV] Fold {fold}")

        # Split
        X_cat_train = X_c.iloc[train_idx]
        X_cat_val   = X_c.iloc[val_idx]
        X_lgbm_train = X_l.iloc[train_idx]
        X_lgbm_val   = X_l.iloc[val_idx]

        y_train_fold = y[train_idx]
        y_val_fold   = y[val_idx]

        # CatBoost model
        model_cat = CatBoostRegressor(
            iterations=iterat,
            learning_rate=learn1,
            depth=dep,
            loss_function=loss_function,
            random_state=rand1,
            verbose=False,
            l2_leaf_reg=leaf_reg,
            random_strength=rand_str,
            bootstrap_type=boot,
            bagging_temperature=bag_temp,
            min_data_in_leaf=min_d,
            early_stopping_rounds=50
        )

        # LGBM model
        model_lgbm = LGBMRegressor(
            n_estimators=n_est,
            learning_rate=learn2,
            max_depth=max_d,
            subsample=sub,
            colsample_bytree=col,
            random_state=rand2,
            verbose=-1,
            stopping_rounds=50
        )

        # Fit
        model_cat.fit(
            X_cat_train,
            y_train_fold,
            cat_features=cat_feature_indices,
        )
        model_lgbm.fit(X_lgbm_train, y_train_fold)

        # Predict + mean ensemble
        p1 = np.ravel(model_cat.predict(X_cat_val))
        p2 = np.ravel(model_lgbm.predict(X_lgbm_val))
        y_pred = (p1 + p2) / 2.0

        oof_preds[val_idx] = y_pred

        # Metrics
        r2   = r2_score(y_val_fold, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        mae  = mean_absolute_error(y_val_fold, y_pred)
        rho, p_value = spearmanr(y_val_fold, y_pred)
        if binary: AUROC = roc_auc_score(y_val_fold, y_pred)

        r2_scores.append(r2)
        rmses.append(rmse)
        maes.append(mae)
        rhos.append(rho)
        pvals.append(p_value)
        if binary: AUC.append(AUROC)

        if verbose:
            if binary: print(f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, Rho={rho:.3f}, pval={p_value:.3e}, AUROC={AUROC:.3f}")
            else: print(f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, Rho={rho:.3f}, pval={p_value:.3e}")

        models.append({"cat": model_cat, "lgbm": model_lgbm})

    if verbose:
        print("\n[Ensemble CV] Overall:")
        print(f"R2   mean={np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
        print(f"RMSE mean={np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
        print(f"MAE  mean={np.mean(maes):.3f} ± {np.std(maes):.3f}")
        print(f"Rho  mean={np.mean(rhos):.3f} ± {np.std(rhos):.3f}")
        if binary: print(f"AUROC mean={np.mean(AUC):.3f} ± {np.std(AUC):.3f}")

    return {
        "models": models,
        "oof_preds": oof_preds,
        "r2_scores": r2_scores,
        "rmses": rmses,
        "maes": maes,
        "rhos": rhos,
        "AUROC": AUC,
        "feature_cols_cat": feature_cols_cat,
        "feature_cols_lgbm": feature_cols_lgbm,
        "cat_cols": cat_cols,
        "group_col": group_col,
    }




def tune_gpb_with_optuna(
    cat_df,
    target_col,
    cat_cols, 
    n_trials=50,
    cv_type="kfold",
    group_col=None,
    n_splits=5,
):
    """
    Run Optuna to tune cv_regression_ensemble.
    All data-related objects are passed explicitly to avoid scope issues.
    """

    def objective(trial):
       
        # --- search space cat ---
        iterat    = trial.suggest_int("iterat", 300, 6500)
        learn1    = trial.suggest_float("learn", 0.01, 0.2, log=True)
        dep       = trial.suggest_int("dep", 2, 10)
        leaf_reg  = trial.suggest_float("leaf_reg", 1.0, 30.0, log=True)
        rand_str  = trial.suggest_float("rand_str", 0.0, 2.0)
        bag_temp  = trial.suggest_float("bag_temp", 0.2, 2.0)
        min_d     = trial.suggest_int("min_d", 5, 60)
        boot      = "Bayesian"

         # --- search space lgbm ---
        n_est = trial.suggest_int("n_est", 300, 4000)                         # trees
        learn2 = trial.suggest_float("learn", 0.01, 0.2, log=True)             # learning_rate
        max_d = trial.suggest_categorical("max_d", [-1, 3, 5, 7, 9, 12])      # max_depth
        sub   = trial.suggest_float("sub", 0.5, 1.0)                           # subsample
        col   = trial.suggest_float("col", 0.3, 1.0)                           # colsample_bytree

        rand1 = 42  # fixed seed for reproducibility
        rand2 = 42  # fixed seed for reproducibility
        

        results_ens = ensemble(
            catdf=cat_df,
            lgbmdf=lgbm_df,
            target_col=target_col,
            cat_cols=cat_cols,
            group_col=group_col,
            n_splits=5,
            # CatBoost
            iterat=iterat, learn1=learn1, dep=dep, rand1=rand1,
            leaf_reg=leaf_reg, rand_str=rand_str, bag_temp=bag_temp, min_d=min_d, boot="Bayesian",
            # LGBM
            n_est=n_est, learn2=learn2, max_d=max_d, sub=sub, col=col, rand2=rand2,
            # Verbosity
            verbose=True
    
        )

        r2_mean = float(np.mean(results_ens["r2_scores"]))
        # Optuna will MAXIMIZE this
        return r2_mean

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best r2:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study
    