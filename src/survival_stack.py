#!/usr/bin/env python3
# Survival-native trainer for QRT (IPCW-C @ 7y) with CENTER-grouped CV, HPO, and stacking.
# Usage examples:
#   python src/survival_stack.py --data-root data --out submissions/submission_survival.csv
#   python src/survival_stack.py --data-root .    --out submissions/submission_survival.csv
import argparse, os, re, sys
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge

# survival-native
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

RNG = int(os.environ.get("SEED", 2025))
np.random.seed(RNG)

CYTO_PATTERNS = {
    "cyto_monosomy7": r"(?:^|[,;\s])-7(?!\d)",
    "cyto_del5q":     r"del\(5q\)",
    "cyto_trisomy8":  r"\+8(?!\d)",
    "cyto_complex":   r"complex|(?:\()?\d+\s*abn\."
}

def featurize_cytogenetics(s: pd.Series) -> pd.DataFrame:
    s = s.fillna("")
    out = pd.DataFrame(index=s.index)
    out["cyto_sex_xx"] = s.str.contains(r"46,\s*XX", case=False, regex=True).astype(int)
    out["cyto_sex_xy"] = s.str.contains(r"46,\s*XY", case=False, regex=True).astype(int)
    for name, rgx in CYTO_PATTERNS.items():
        out[name] = s.str.contains(rgx, case=True, regex=True).astype(int)
    out["cyto_abn_sepcount"] = s.str.count(r"[,;]").fillna(0)
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
        self.effects_ = list(d.get("EFFECT", pd.Series([], dtype=str)).astype(str).fillna("unknown").value_counts().head(self.top_effects).index)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["GENE"] = d["GENE"].astype(str).fillna("unknown")
        if "VAF" in d.columns:
            d["VAF"] = pd.to_numeric(d["VAF"], errors="coerce")
        else:
            d["VAF"] = np.nan
        gene = (
            d[d["GENE"].isin(self.genes_)]
            .assign(present=1)
            .drop_duplicates(["ID","GENE"])  # one per gene/patient
            .pivot_table(index="ID", columns="GENE", values="present", aggfunc="max", fill_value=0)
            .reindex(columns=self.genes_, fill_value=0)
            .add_prefix("gene_")
        )
        if self.effects_ and "EFFECT" in d.columns:
            eff = (
                d[d["EFFECT"].isin(self.effects_)]
                .groupby(["ID","EFFECT"]).size().unstack(fill_value=0)
                .reindex(columns=self.effects_, fill_value=0)
                .add_prefix("effect_")
            )
        else:
            eff = pd.DataFrame(index=d["ID"].drop_duplicates())
        vaf = d.groupby("ID").agg(mut_count=("GENE","count"), vaf_mean=("VAF","mean"), vaf_max=("VAF","max"), vaf_std=("VAF","std")).fillna(0)
        return gene.join([eff, vaf], how="outer").fillna(0).reset_index()


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    cols = [c for c in df.columns if c != "ID"]
    num_cols = [c for c in cols if df[c].dtype.kind in "fcui" and c != "CENTER"]
    cat_cols = [c for c in cols if df[c].dtype == object and c in {"CENTER"}]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ], remainder="drop", verbose_feature_names_out=False)


def surv_from_df(y: pd.DataFrame):
    y2 = y[["OS_YEARS","OS_STATUS"]].copy()
    y2["OS_YEARS"] = pd.to_numeric(y2["OS_YEARS"], errors="coerce")
    y2["OS_STATUS"] = pd.to_numeric(y2["OS_STATUS"], errors="coerce").round().clip(0,1)
    mask = y2["OS_YEARS"].notna() & y2["OS_STATUS"].isin([0,1])
    return Surv.from_dataframe(event="OS_STATUS", time="OS_YEARS", data=y2.loc[mask]), mask


def grid_run(name: str, pipe: Pipeline, grid: Dict[str, List], X, y_surv_all, mask, groups, tau, Xte) -> Tuple[str, Dict, np.ndarray, np.ndarray, List[float]]:
    X = X.reset_index(drop=True)
    Xte = Xte.reset_index(drop=True)
    y = y_surv_all[mask]
    groups = groups.reset_index(drop=True)[mask.reset_index(drop=True)]

    uniq_groups = groups.unique()
    n_splits = min(5, max(2, len(uniq_groups)))
    gkf = GroupKFold(n_splits=n_splits)

    keys = list(grid.keys()); values = [grid[k] for k in keys]
    best_mean = -np.inf; best_params = {}; best_oof = None; best_test = None; best_scores: List[float] = []

    for combo in product(*values):
        params = dict(zip(keys, combo))
        model = pipe.set_params(**params)
        oof = np.zeros(len(X))
        test_folds = []
        fold_scores: List[float] = []
        for tr_idx, va_idx in gkf.split(X, groups=groups):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            model.fit(X_tr, y_tr)
            r_va = model.predict(X_va)
            sc = float(concordance_index_ipcw(y_tr, y_va, r_va, tau=tau)[0])
            fold_scores.append(sc)
            oof[va_idx] = r_va
            test_folds.append(model.predict(Xte))
        mean_sc = float(np.mean(fold_scores))
        if mean_sc > best_mean:
            best_mean = mean_sc; best_params = params
            best_oof = oof; best_test = np.mean(np.vstack(test_folds), axis=0)
            best_scores = fold_scores
        print(f"[HPO] {name} {params} -> C@{tau}={mean_sc:.4f}")
    print(f"[BEST] {name} {best_params} meanC={best_mean:.4f}")
    return name, best_params, best_oof, best_test, best_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="submissions/submission_survival.csv")
    ap.add_argument("--top-genes", type=int, default=1000)
    ap.add_argument("--top-effects", type=int, default=48)
    ap.add_argument("--tau", type=float, default=7.0)
    args = ap.parse_args()

    req = {
        "clinical_train": os.path.join(args.data_root, "clinical_train.csv"),
        "molecular_train": os.path.join(args.data_root, "molecular_train.csv"),
        "target_train": os.path.join(args.data_root, "target_train.csv"),
        "clinical_test": os.path.join(args.data_root, "clinical_test.csv"),
        "molecular_test": os.path.join(args.data_root, "molecular_test.csv"),
    }
    for k,p in req.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {k}: {p}. Use --data-root . if files are at repo root.")

    clin_tr = pd.read_csv(req["clinical_train"]) ; clin_te = pd.read_csv(req["clinical_test"]) 
    y_tr    = pd.read_csv(req["target_train"])  
    mol_tr  = pd.read_csv(req["molecular_train"]) ; mol_te = pd.read_csv(req["molecular_test"]) 

    agg = MolAgg(top_genes=args.top_genes, top_effects=args.top_effects).fit(mol_tr)
    mol_tr_f = agg.transform(mol_tr); mol_te_f = agg.transform(mol_te)

    X_train = clin_tr.merge(mol_tr_f, on="ID", how="left")
    X_test  = clin_te.merge(mol_te_f, on="ID", how="left")

    if "CYTOGENETICS" in X_train.columns:
        X_train = pd.concat([X_train, featurize_cytogenetics(X_train["CYTOGENETICS"])], axis=1)
    if "CYTOGENETICS" in X_test.columns:
        X_test = pd.concat([X_test, featurize_cytogenetics(X_test["CYTOGENETICS"])], axis=1)

    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[X_train.columns]

    y_surv_all, mask_all = surv_from_df(y_tr)
    Xy = X_train.merge(y_tr, on="ID", how="left")
    mask = Xy["OS_YEARS"].notna() & Xy["OS_STATUS"].isin([0,1])
    X_train = X_train.loc[mask].reset_index(drop=True)
    y_surv = y_surv_all[mask_all & mask]

    groups = X_train["CENTER"].astype(str) if "CENTER" in X_train.columns else pd.Series(["0"]*len(X_train))

    feat_cols = [c for c in X_train.columns if c != "ID"]
    Xtr = X_train[feat_cols].reset_index(drop=True)
    Xte = X_test[feat_cols].reset_index(drop=True)

    pre = build_preprocessor(X_train)

    cox = Pipeline([("pre", pre), ("clf", CoxnetSurvivalAnalysis(alphas=10 ** np.linspace(-3, 1, 36), l1_ratio=0.5, fit_baseline_model=True, random_state=RNG))])
    rsf = Pipeline([("pre", pre), ("clf", RandomSurvivalForest(n_estimators=800, min_samples_leaf=3, max_features="sqrt", random_state=RNG, n_jobs=-1))])
    gbs = Pipeline([("pre", pre), ("clf", GradientBoostingSurvivalAnalysis(loss="coxph", learning_rate=0.05, n_estimators=700, max_depth=2, random_state=RNG))])

    tau = float(args.tau)
    specs = [
        ("coxnet", cox, {"clf__l1_ratio": [0.3, 0.5, 0.8]}),
        ("rsf",    rsf, {"clf__n_estimators": [600, 800], "clf__min_samples_leaf": [3,5], "clf__max_features": ["sqrt", 0.3]}),
        ("gbsa",   gbs, {"clf__n_estimators": [500, 700], "clf__max_depth": [2,3] , "clf__learning_rate": [0.03, 0.05]}),
    ]

    oof = {}; te = {}; scores = {}
    for name, pipe, grid in specs:
        nm, params, o, t, sc = grid_run(name, pipe, grid, Xtr, y_surv_all, mask, groups, tau, Xte)
        oof[nm] = o; te[nm] = t; scores[nm] = sc

    # stack on OOF risks
    Ztr = np.column_stack([oof[k] for k in ["coxnet","rsf","gbsa"] if k in oof])
    Zte = np.column_stack([te[k]  for k in ["coxnet","rsf","gbsa"] if k in te])
    # pseudo target = negative time (only for regression fit, we evaluate with IPCW)
    y_df = y_tr.loc[mask, ["OS_YEARS","OS_STATUS"]]
    T = y_df["OS_YEARS"].to_numpy(); E = y_df["OS_STATUS"].to_numpy()
    y_pseudo = -np.minimum(T, tau); y_pseudo[E==0] *= 0.6
    stk = Pipeline([("sc", StandardScaler()), ("lr", Ridge(alpha=1.0, random_state=RNG))])
    stk.fit(Ztr, y_pseudo)
    oof_st = stk.predict(Ztr)

    # final oof and test via rank average
    def rank(x): return pd.Series(x).rank(method="average").to_numpy()
    oof_rank = np.mean(np.column_stack([rank(oof[k]) for k in oof] + [rank(oof_st)]), axis=1)
    te_rank  = np.mean(np.column_stack([rank(te[k])  for k in te]  + [rank(stk.predict(Zte))]), axis=1)

    c = float(concordance_index_ipcw(y_surv, y_surv, oof_rank, tau=tau)[0])
    print(f"[STACK] OOF C@{tau} = {c:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sub = pd.DataFrame({"risk_score": te_rank.astype(float)}, index=clin_te.set_index("ID").index)
    sub.index.name = "ID"
    sub.to_csv(args.out)
    print(f"[DONE] Wrote {args.out}  rows={len(sub)}")

if __name__ == "__main__":
    main()
