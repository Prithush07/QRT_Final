#!/usr/bin/env python3
"""
Survival-native trainer for QRT (official IPCW-C @ 7y) with CENTER-grouped CV,
small HPO, rank blend + stacking. Now with LightGBM & XGBoost add-ons
(trained as censored-aware pseudo-regressors), while Coxnet/RSF/GBSA remain
survival-native.

Usage:
  python src/survival_stack_fixed_lgbm_xgb.py --data-root data --out submissions/submission_survival.csv \
    --top-genes 1000 --top-effects 48 --tau 7 --mode quick --seeds 2025,1337

Install (if needed):
  pip install scikit-survival lightgbm xgboost
"""
import argparse
import os
import inspect
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

# Optional boosters (auto-skip if not installed)
_HAS_LGBM = False
try:
    import lightgbm as lgb  # type: ignore
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

_HAS_XGB = False
try:
    import xgboost as xgb  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

RNG = int(os.environ.get("SEED", 2025))
np.random.seed(RNG)
VERSION = "v2.2-with-lgbm-xgb-2025-08-31"

# -------------------------------
# Helpers
# -------------------------------

def make_ohe():
    """Create OneHotEncoder compatible with sklearn versions."""
    params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def rank_avg(arrs: List[np.ndarray]) -> np.ndarray:
    ranks = [pd.Series(a).rank(method="average").to_numpy() for a in arrs]
    return np.mean(ranks, axis=0)

# -------------------------------
# Feature engineering
# -------------------------------
CYTO_PATTERNS = {
    "cyto_monosomy7": r"(?:^|[,;\s])-7(?!\d)|del\(7q\)",
    "cyto_del5q": r"del\(5q\)|-5(?!\d)",
    "cyto_trisomy8": r"\+8(?!\d)",
    "cyto_complex": r"complex|(?:\()?\d+\s*abn\.",
}


def featurize_cytogenetics(s: pd.Series) -> pd.DataFrame:
    s = s.fillna("")
    out = pd.DataFrame(index=s.index)
    out["cyto_sex_xx"] = s.str.contains(r"46,\s*XX", case=False, regex=True).astype(int)
    out["cyto_sex_xy"] = s.str.contains(r"46,\s*XY", case=False, regex=True).astype(int)
    for name, rgx in CYTO_PATTERNS.items():
        out[name] = s.str.contains(rgx, case=True, regex=True).astype(int)
    out["cyto_abn_sepcount"] = s.str.count(r"[,;]").fillna(0)
    out["cyto_complex_ge3"] = (out["cyto_abn_sepcount"] >= 2).astype(int)
    out["cyto_mk"] = s.str.contains(r"(?:^|[,;\s])-(?:[1-9]|1\d|2\d)(?:[,;\s]-)", regex=True).astype(int)
    out["cyto_missing"] = (s == "").astype(int)
    return out


@dataclass
class MolAgg:
    top_genes: int = 1000
    top_effects: int = 48
    genes_: List[str] = None
    effects_: List[str] = None

    def fit(self, df: pd.DataFrame):
        d = df.copy()
        d["GENE"] = d["GENE"].astype(str).fillna("unknown")
        self.genes_ = list(d["GENE"].value_counts().head(self.top_genes).index)
        self.effects_ = list(
            d.get("EFFECT", pd.Series([], dtype=str))
            .astype(str)
            .fillna("unknown")
            .value_counts()
            .head(self.top_effects)
            .index
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["GENE"] = d["GENE"].astype(str).fillna("unknown")
        d["VAF"] = pd.to_numeric(d.get("VAF", np.nan), errors="coerce")
        gene = (
            d[d["GENE"].isin(self.genes_)]
            .assign(present=1)
            .drop_duplicates(["ID", "GENE"])
            .pivot_table(index="ID", columns="GENE", values="present", aggfunc="max", fill_value=0)
            .reindex(columns=self.genes_, fill_value=0)
            .add_prefix("gene_")
        )
        if self.effects_ and "EFFECT" in d.columns:
            eff = (
                d[d["EFFECT"].isin(self.effects_)]
                .groupby(["ID", "EFFECT"]).size().unstack(fill_value=0)
                .reindex(columns=self.effects_, fill_value=0)
                .add_prefix("effect_")
            )
        else:
            eff = pd.DataFrame(index=d["ID"].drop_duplicates())
        vaf = (
            d.groupby("ID")
            .agg(mut_count=("GENE", "count"), vaf_mean=("VAF", "mean"), vaf_max=("VAF", "max"), vaf_std=("VAF", "std"))
            .fillna(0)
        )
        out = gene.join([eff, vaf], how="outer").fillna(0).reset_index()
        for genes, name in [
            (["DNMT3A","TET2","IDH1","IDH2"],"bucket_methyl"),
            (["SRSF2","SF3B1","U2AF1","ZRSR2"],"bucket_splice"),
            (["FLT3","NRAS","KRAS","KIT","PTPN11"],"bucket_signal"),
            (["ASXL1","EZH2"],"bucket_chrom"),
            (["TP53","WT1"],"bucket_tumor"),
            (["RUNX1","CEBPA"],"bucket_tx"),
        ]:
            out[name] = 0
            for g in genes:
                col = "gene_" + g
                if col in out.columns:
                    out[name] = out[name] | (out[col] > 0).astype(int)
        for g in ["NPM1", "FLT3", "TP53", "RUNX1", "CEBPA"]:
            col = "gene_" + g
            if col in out.columns:
                out[g + "_VAFx"] = out[col] * out.get("vaf_mean", 0.0)
        return out


# -------------------------------
# Preprocessing
# -------------------------------

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    cols = [c for c in df.columns if c != "ID"]
    num_cols = [c for c in cols if df[c].dtype.kind in "fcui" and c != "CENTER"]
    cat_cols = [c for c in cols if df[c].dtype == object and c in {"CENTER"}]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="MISSING")), ("oh", make_ohe())]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# -------------------------------
# Grid runners
# -------------------------------

def grid_run_surv(name: str, pipe: Pipeline, grid: Dict[str, List], X: pd.DataFrame, y_surv: np.ndarray, groups: pd.Series, tau: float, Xte: pd.DataFrame, folds: int) -> Tuple[str, Dict, np.ndarray, np.ndarray, List[float]]:
    gkf = GroupKFold(n_splits=folds)
    keys = list(grid.keys()); values = [grid[k] for k in keys]
    best_mean, best_params, best_oof, best_test, best_scores = -np.inf, {}, None, None, []
    for combo in product(*values):
        params = dict(zip(keys, combo))
        model = pipe.set_params(**params)
        if any(isinstance(s, CoxnetSurvivalAnalysis) for _, s in model.steps):
            try:
                model.set_params(clf__alphas=10 ** np.linspace(-2, 1, 24), clf__max_iter=200000, clf__tol=1e-7)
            except ValueError:
                model.set_params(clf__alphas=10 ** np.linspace(-2, 1, 24))
        oof = np.zeros(len(X)); test_folds = []; fold_scores: List[float] = []
        for tr, va in gkf.split(X, groups=groups):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y_surv[tr], y_surv[va]
            tries = 0
            while True:
                try:
                    model.fit(X_tr, y_tr)
                    break
                except ArithmeticError:
                    if any(isinstance(s, CoxnetSurvivalAnalysis) for _, s in model.steps) and tries < 3:
                        alphas = model.get_params().get("clf__alphas") or 10 ** np.linspace(-2, 1, 24)
                        model.set_params(clf__alphas=np.unique(np.clip(np.array(alphas) * 10.0, 1e-3, 1e3)))
                        tries += 1
                        print(f"[WARN] Coxnet numerical -> raise alphas (try {tries})")
                        continue
                    else:
                        raise
            r_va = model.predict(X_va)
            sc = float(concordance_index_ipcw(y_tr, y_va, r_va, tau=tau)[0])
            fold_scores.append(sc)
            oof[va] = r_va
            test_folds.append(model.predict(Xte))
        mean_sc = float(np.mean(fold_scores))
        print(f"[HPO] {name} {params} -> C@{tau}={mean_sc:.4f}")
        if mean_sc > best_mean:
            best_mean, best_params, best_oof, best_test, best_scores = mean_sc, params, oof, np.mean(np.vstack(test_folds), axis=0), fold_scores
    print(f"[BEST] {name} {best_params} meanC={best_mean:.4f}")
    return name, best_params, best_oof, best_test, best_scores


def grid_run_reg(name: str, pipe: Pipeline, grid: Dict[str, List], X: pd.DataFrame, y_surv: np.ndarray, groups: pd.Series, tau: float, Xte: pd.DataFrame, folds: int) -> Tuple[str, Dict, np.ndarray, np.ndarray, List[float]]:
    """For LGBM/XGB trained on pseudo target y_pseudo = -min(time, tau), with 0.6 weight for censored."""
    gkf = GroupKFold(n_splits=folds)
    keys = list(grid.keys()); values = [grid[k] for k in keys]
    best_mean, best_params, best_oof, best_test, best_scores = -np.inf, {}, None, None, []
    for combo in product(*values):
        params = dict(zip(keys, combo))
        model = pipe.set_params(**params)
        oof = np.zeros(len(X)); test_folds = []; fold_scores: List[float] = []
        for tr, va in gkf.split(X, groups=groups):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y_surv[tr], y_surv[va]
            T = y_tr['time'].astype(float); E = y_tr['event'].astype(int)
            y_pseudo = -np.minimum(T, tau); y_pseudo[E == 0] *= 0.6
            model.fit(X_tr, y_pseudo)
            r_va = model.predict(X_va)
            sc = float(concordance_index_ipcw(y_tr, y_va, r_va, tau=tau)[0])
            fold_scores.append(sc)
            oof[va] = r_va
            test_folds.append(model.predict(Xte))
        mean_sc = float(np.mean(fold_scores))
        print(f"[HPO] {name} {params} -> C@{tau}={mean_sc:.4f}")
        if mean_sc > best_mean:
            best_mean, best_params, best_oof, best_test, best_scores = mean_sc, params, oof, np.mean(np.vstack(test_folds), axis=0), fold_scores
    print(f"[BEST] {name} {best_params} meanC={best_mean:.4f}")
    return name, best_params, best_oof, best_test, best_scores


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="submissions/submission_survival.csv")
    ap.add_argument("--top-genes", type=int, default=1000)
    ap.add_argument("--top-effects", type=int, default=48)
    ap.add_argument("--tau", type=float, default=7.0)
    ap.add_argument("--seeds", type=str, default="2025")
    ap.add_argument("--mode", choices=["quick","max"], default="quick")
    args = ap.parse_args()

    print(f"[survival_stack] VERSION {VERSION} | LGBM={_HAS_LGBM} XGB={_HAS_XGB} | mode={args.mode}")

    req = {k: os.path.join(args.data_root, f"{k}.csv") for k in ["clinical_train","molecular_train","target_train","clinical_test","molecular_test"]}
    for k,p in req.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {k}: {p}. Use --data-root . if files are at repo root.")

    clin_tr = pd.read_csv(req["clinical_train"]).copy()
    clin_te = pd.read_csv(req["clinical_test"]).copy()
    y_tr = pd.read_csv(req["target_train"]).copy()
    mol_tr = pd.read_csv(req["molecular_train"]).copy()
    mol_te = pd.read_csv(req["molecular_test"]).copy()

    # Molecular agg
    agg = MolAgg(top_genes=args.top_genes, top_effects=args.top_effects).fit(mol_tr)
    mol_tr_f = agg.transform(mol_tr)
    mol_te_f = agg.transform(mol_te)

    # Merge features
    X_train = clin_tr.merge(mol_tr_f, on="ID", how="left")
    X_test  = clin_te.merge(mol_te_f, on="ID", how="left")

    # Cytogenetics
    if "CYTOGENETICS" in X_train.columns:
        X_train = pd.concat([X_train, featurize_cytogenetics(X_train["CYTOGENETICS"])], axis=1)
    if "CYTOGENETICS" in X_test.columns:
        X_test  = pd.concat([X_test,  featurize_cytogenetics(X_test["CYTOGENETICS"])], axis=1)

    # Ensure test has same columns as train
    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[X_train.columns]

    # Align target with merged design matrix and build Surv
    Xy = X_train.merge(y_tr, on="ID", how="left")
    Xy["OS_YEARS"] = pd.to_numeric(Xy["OS_YEARS"], errors="coerce")
    Xy["OS_STATUS"] = pd.to_numeric(Xy["OS_STATUS"], errors="coerce").round().clip(0, 1)
    mask = Xy["OS_YEARS"].notna() & Xy["OS_STATUS"].isin([0, 1])
    dropped = int((~mask).sum())
    if dropped:
        print(f"[CLEAN] Dropped {dropped} rows with missing/invalid OS labels.")
    X_train = X_train.loc[mask].reset_index(drop=True)
    Xy = Xy.loc[mask].reset_index(drop=True)
    y_surv = Surv.from_dataframe(event="OS_STATUS", time="OS_YEARS", data=Xy)

    # Groups after filtering
    groups = X_train.get("CENTER", pd.Series(["0"]*len(X_train))).astype(str).reset_index(drop=True)

    # Design matrices
    feat_cols = [c for c in X_train.columns if c != "ID"]
    Xtr = X_train[feat_cols].reset_index(drop=True)
    Xte = X_test[feat_cols].reset_index(drop=True)

    # Preprocessor
    pre = build_preprocessor(X_train)

    # Models & grids
    if args.mode == "quick":
        folds = 3
        rsf_grid = {"clf__n_estimators": [600, 800], "clf__min_samples_leaf": [3, 5], "clf__max_features": ["sqrt", 0.3]}
        gbs_grid = {"clf__n_estimators": [500, 700], "clf__max_depth": [2, 3], "clf__learning_rate": [0.03, 0.05]}
        cox_grid = {"clf__l1_ratio": [0.3, 0.5]}
        lgb_grid = {"clf__n_estimators": [800], "clf__num_leaves": [31, 63], "clf__learning_rate": [0.05]}
        xgb_grid = {"clf__n_estimators": [900], "clf__max_depth": [3], "clf__learning_rate": [0.05]}
    else:
        folds = 5
        rsf_grid = {"clf__n_estimators": [800, 1200], "clf__min_samples_leaf": [2, 4, 6], "clf__max_features": ["sqrt", 0.3]}
        gbs_grid = {"clf__n_estimators": [700, 1100], "clf__max_depth": [2, 3], "clf__learning_rate": [0.03, 0.06]}
        cox_grid = {"clf__l1_ratio": [0.2, 0.5, 0.8]}
        lgb_grid = {"clf__n_estimators": [900, 1300], "clf__num_leaves": [31, 63], "clf__learning_rate": [0.03, 0.06]}
        xgb_grid = {"clf__n_estimators": [1000, 1400], "clf__max_depth": [3, 4], "clf__learning_rate": [0.03, 0.06]}

    cox = Pipeline([("pre", pre), ("clf", CoxnetSurvivalAnalysis(l1_ratio=0.5, fit_baseline_model=False))])
    rsf = Pipeline([("pre", pre), ("clf", RandomSurvivalForest(n_estimators=800, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=RNG))])
    gbs = Pipeline([("pre", pre), ("clf", GradientBoostingSurvivalAnalysis(loss="coxph", learning_rate=0.05, n_estimators=700, max_depth=2, random_state=RNG))])

    specs = [
        ("coxnet", cox, cox_grid, grid_run_surv),
        ("rsf", rsf, rsf_grid, grid_run_surv),
        ("gbsa", gbs, gbs_grid, grid_run_surv),
    ]

    if _HAS_LGBM:
        lgbm = Pipeline([("pre", pre), ("clf", lgb.LGBMRegressor(objective="regression", subsample=0.8, colsample_bytree=0.8, random_state=RNG, n_jobs=-1))])
        specs.append(("lgbm_reg", lgbm, lgb_grid, grid_run_reg))
    if _HAS_XGB:
        xgbr = Pipeline([("pre", pre), ("clf", xgb.XGBRegressor(objective="reg:squarederror", subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=RNG))])
        specs.append(("xgb_reg", xgbr, xgb_grid, grid_run_reg))

    # Multi-seed rank blend
    seed_list = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    oof_seeds: List[np.ndarray] = []
    te_seeds: List[np.ndarray] = []

    for seed in seed_list:
        rsf.set_params(clf__random_state=seed)
        gbs.set_params(clf__random_state=seed)
        if _HAS_LGBM:
            lgbm.set_params(clf__random_state=seed)
        if _HAS_XGB:
            xgbr.set_params(random_state=seed)

        oof: Dict[str, np.ndarray] = {}
        te: Dict[str, np.ndarray] = {}
        for name, pipe, grid, runner in specs:
            nm, params, o, t, sc = runner(name, pipe, grid, Xtr, y_surv, groups, float(args.tau), Xte, folds)
            oof[nm] = o; te[nm] = t

        Ztr = np.column_stack([oof[k] for k in oof])
        Zte = np.column_stack([te[k] for k in te])
        T = Xy["OS_YEARS"].to_numpy(); E = Xy["OS_STATUS"].to_numpy()
        y_pseudo = -np.minimum(T, float(args.tau)); y_pseudo[E == 0] *= 0.6
        stk = Pipeline([("sc", StandardScaler()), ("lr", Ridge(alpha=1.0, random_state=seed))])
        stk.fit(Ztr, y_pseudo)
        oof_st = stk.predict(Ztr)
        oof_rank = rank_avg([oof[k] for k in oof] + [oof_st])
        te_rank  = rank_avg([te[k] for k in te] + [stk.predict(Zte)])
        oof_seeds.append(oof_rank)
        te_seeds.append(te_rank)

    oof_final = rank_avg(oof_seeds) if len(oof_seeds) > 1 else oof_seeds[0]
    te_final  = rank_avg(te_seeds)  if len(te_seeds) > 1 else te_seeds[0]

    c = float(concordance_index_ipcw(y_surv, y_surv, oof_final, tau=float(args.tau))[0])
    print(f"[STACK] OOF C@{args.tau} = {c:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sub = pd.DataFrame({"risk_score": te_final.astype(float)}, index=clin_te.set_index("ID").index)
    sub.index.name = "ID"; sub.to_csv(args.out)
    print(f"[DONE] Wrote {args.out}  rows={len(sub)} | VERSION={VERSION}")


if __name__ == "__main__":
    main()
