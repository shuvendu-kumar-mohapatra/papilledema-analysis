"""Feature correlation, dimensionality reduction (PCA/t-SNE/UMAP), and leak-free ML analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import umap

from src.data.validation import FUNDUS_CLASSES, OCT_3CLASSES, OCT_SEVERITY_CLASSES
from src.feature_extraction.deep_features import DEEP_BACKBONE_DIMS, extract_cnn_embeddings
from src.feature_extraction.handcrafted import extract_handcrafted_dataset
from src.utils.config import load_all_configs, resolve_path


def _assign_contiguous_5fold_oct(df: pd.DataFrame, label_col: str) -> np.ndarray:
    """Assign 5 stratified folds for OCT using prefix groups (>=5) or contiguous slice blocks."""
    folds = np.zeros(len(df), dtype=np.int64)
    df_idx = df.copy().reset_index(drop=True)
    for _, cdf in df_idx.groupby(label_col):
        n_prefixes = cdf["inferred_subject_prefix"].nunique() if "inferred_subject_prefix" in cdf.columns else 0
        if n_prefixes >= 5:
            pref_sizes = cdf.groupby("inferred_subject_prefix").size().sort_values(ascending=False)
            fold_counts = np.zeros(5, dtype=np.int64)
            pref_map: dict[str, int] = {}
            for p, sz in pref_sizes.items():
                f_idx = int(np.argmin(fold_counts))
                pref_map[str(p)] = f_idx
                fold_counts[f_idx] += int(sz)
            folds[cdf.index.to_numpy()] = cdf["inferred_subject_prefix"].map(pref_map).to_numpy()
        else:
            sort_cols = [c for c in ("inferred_sequence_id", "slice_index", "image_path") if c in cdf.columns]
            ordered = cdf.sort_values(sort_cols) if sort_cols else cdf
            chunks = np.array_split(ordered.index.to_numpy(), 5)
            for f_idx, idx_arr in enumerate(chunks):
                folds[idx_arr] = f_idx
    return folds


def _make_leakfree_5fold_indices(df: pd.DataFrame, track: str, label_col: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Build 5 train/test fold index pairs for leak-free handcrafted feature classifier evaluation."""
    if track == "fundus_3class" and "cluster_id" in df.columns:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(sgkf.split(df, df[label_col], df["cluster_id"]))
        return splits, "StratifiedGroupKFold (cluster_id)"
    folds = _assign_contiguous_5fold_oct(df, label_col)
    all_idx = np.arange(len(df))
    splits = [(all_idx[folds != f], all_idx[folds == f]) for f in range(5)]
    return splits, "Contiguous Slice-Block + Prefix Balanced 5-Fold (NOT patient-independent)"


def _run_dim_reduction(
    X: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute standardized PCA variance curve + 2D PCA, t-SNE, and UMAP projections (exploratory)."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
    n_comp = min(50, Xs.shape[1], max(2, Xs.shape[0] - 1))
    pca_full = PCA(n_components=n_comp, random_state=seed)
    X_pca = pca_full.fit_transform(Xs)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    perp = float(min(30, max(5, (len(Xs) - 1) // 4)))
    tsne = TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", random_state=seed)
    X_tsne = tsne.fit_transform(X_pca[:, : min(30, X_pca.shape[1])])

    reducer = umap.UMAP(n_components=2, n_neighbors=min(15, max(5, len(Xs) - 1)), min_dist=0.1, random_state=seed)
    X_umap = reducer.fit_transform(X_pca[:, : min(30, X_pca.shape[1])])
    return X_pca[:, :2], X_tsne, X_umap, cum_var


def _evaluate_handcrafted_ml(
    df: pd.DataFrame,
    X: np.ndarray,
    feat_names: list[str],
    track: str,
    label_col: str,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Evaluate leak-free RandomForest & LogisticRegression inside 5-fold CV and extract feature importance."""
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str))
    splits, cv_scheme = _make_leakfree_5fold_indices(df, track, label_col)

    clf_rows: list[dict[str, Any]] = []
    imp_accum = np.zeros(len(feat_names), dtype=np.float64)

    for clf_name, estimator in (
        ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)),
    ):
        oof_pred = np.zeros_like(y)
        fold_f1s: list[float] = []
        fold_accs: list[float] = []
        fold_bals: list[float] = []
        for f_idx, (tr_idx, te_idx) in enumerate(splits):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", estimator.__class__(**estimator.get_params())),
            ])
            pipe.fit(X[tr_idx], y[tr_idx])
            pred = pipe.predict(X[te_idx])
            oof_pred[te_idx] = pred
            fold_f1s.append(float(f1_score(y[te_idx], pred, average="macro", zero_division=0)))
            fold_accs.append(float(accuracy_score(y[te_idx], pred)))
            fold_bals.append(float(balanced_accuracy_score(y[te_idx], pred)))
            if clf_name == "RandomForest":
                imp_accum += pipe.named_steps["clf"].feature_importances_ / len(splits)

        clf_rows.append({
            "track": track,
            "feature_set": f"Handcrafted ({len(feat_names)} scalar features)",
            "classifier": clf_name,
            "cv_scheme": cv_scheme,
            "mean_fold_macro_f1": float(np.mean(fold_f1s)),
            "std_fold_macro_f1": float(np.std(fold_f1s, ddof=1)),
            "oof_macro_f1": float(f1_score(y, oof_pred, average="macro", zero_division=0)),
            "oof_accuracy": float(accuracy_score(y, oof_pred)),
            "oof_balanced_accuracy": float(balanced_accuracy_score(y, oof_pred)),
        })

    # Univariate Kruskal-Wallis discriminative H-statistic across full dataset (descriptive)
    kw_rows = []
    for j, fname in enumerate(feat_names):
        groups = [X[y == c, j] for c in np.unique(y)]
        try:
            h_stat, p_val = stats.kruskal(*groups)
        except Exception:
            h_stat, p_val = 0.0, 1.0
        kw_rows.append({
            "track": track,
            "feature": fname,
            "rf_mean_importance": float(imp_accum[j]),
            "kruskal_h": float(0.0 if np.isnan(h_stat) else h_stat),
            "kruskal_p": float(1.0 if np.isnan(p_val) else p_val),
        })
    imp_df = pd.DataFrame(kw_rows).sort_values("rf_mean_importance", ascending=False).reset_index(drop=True)
    return clf_rows, imp_df


def run_feature_extraction_pipeline(
    cfg: dict[str, Any] | None = None,
    output_dir: str | Path = "results/feature_extraction",
) -> dict[str, Any]:
    """Execute Phase 3 handcrafted + deep feature extraction, DR, correlation & ML analysis."""
    from src.feature_extraction.plots import generate_all_feature_figures

    cfg = cfg or load_all_configs()
    out_root = resolve_path(output_dir)
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = resolve_path(cfg["paths"]["splits_dir"])
    ckpt_dir = resolve_path(cfg["paths"]["checkpoints_dir"])

    fundus_df = pd.read_csv(splits_dir / "fundus_splits.csv")
    oct_3_df = pd.read_csv(splits_dir / "oct_3class_splits.csv")
    oct_sev_df = pd.read_csv(splits_dir / "oct_severity_splits.csv")

    f_cfg = cfg["fundus"]
    o_cfg = cfg["oct"]

    # 1. Handcrafted Extraction (Separate Fundus & OCT)
    f_hc_df, f_hog = extract_handcrafted_dataset(fundus_df, modality="fundus", image_size=int(f_cfg["image_size"]))
    o_hc_df, o_hog = extract_handcrafted_dataset(oct_3_df, modality="oct", image_size=int(o_cfg["image_size"]))

    f_meta_cols = {"sample_id", "image_path", "label", "split", "cluster_id"}
    f_feat_cols = [c for c in f_hc_df.columns if c not in f_meta_cols]
    o_meta_cols = {"sample_id", "image_path", "label", "split", "label_severity", "inferred_subject_prefix", "inferred_sequence_id"}
    o_feat_cols = [c for c in o_hc_df.columns if c not in o_meta_cols]

    # Severity subset aligns directly with oct_3_df mask
    osv_hc_df = o_hc_df[o_hc_df["label_severity"].notna()].copy().reset_index(drop=True)
    osv_hc_df["label"] = osv_hc_df["label_severity"].astype(str)
    osv_mask = o_hc_df["label_severity"].notna().to_numpy()
    osv_hog = o_hog[osv_mask]

    f_hc_df.to_csv(out_root / "fundus_handcrafted_features.csv", index=False)
    o_hc_df.to_csv(out_root / "oct_handcrafted_features.csv", index=False)
    osv_hc_df.to_csv(out_root / "oct_severity_handcrafted_features.csv", index=False)

    np.savez_compressed(
        out_root / "fundus_handcrafted.npz",
        X_scalar=f_hc_df[f_feat_cols].to_numpy(dtype=np.float32),
        X_hog=f_hog,
        feature_names=np.asarray(f_feat_cols),
        sample_id=f_hc_df["sample_id"].to_numpy(),
        label=f_hc_df["label"].to_numpy(),
        split=f_hc_df["split"].to_numpy(),
    )
    np.savez_compressed(
        out_root / "oct_handcrafted.npz",
        X_scalar=o_hc_df[o_feat_cols].to_numpy(dtype=np.float32),
        X_hog=o_hog,
        feature_names=np.asarray(o_feat_cols),
        sample_id=o_hc_df["sample_id"].to_numpy(),
        label=o_hc_df["label"].to_numpy(),
        label_severity=o_hc_df["label_severity"].fillna("").to_numpy(),
        split=o_hc_df["split"].to_numpy(),
    )

    # 2. Standalone Deep Feature Extraction from existing seed-42 checkpoints
    track_specs = [
        ("fundus_3class", fundus_df, "fundus", 3, f_cfg, "label", FUNDUS_CLASSES, f_hc_df[f_feat_cols].to_numpy(dtype=np.float32)),
        ("oct_individual", oct_3_df, "oct", 3, o_cfg, "label", OCT_3CLASSES, o_hc_df[o_feat_cols].to_numpy(dtype=np.float32)),
        ("oct_severity", oct_sev_df, "oct", 4, o_cfg, "label", OCT_SEVERITY_CLASSES, osv_hc_df[o_feat_cols].to_numpy(dtype=np.float32)),
    ]

    deep_summary_rows: list[dict[str, Any]] = []
    pca_var_rows: list[dict[str, Any]] = []
    dr_tables: dict[str, pd.DataFrame] = {}

    for track_name, df_trk, modality, n_cls, m_cfg, lbl_col, cls_order, X_hc_mat in track_specs:
        emb_dict: dict[str, np.ndarray] = {}
        for bb_name in DEEP_BACKBONE_DIMS:
            ckpt_p = ckpt_dir / f"{track_name}__{bb_name}__seed42" / "best.pt"
            mat, meta = extract_cnn_embeddings(
                df=df_trk,
                modality=modality,
                backbone_name=bb_name,
                num_classes=n_cls,
                checkpoint_path=ckpt_p,
                image_size=int(m_cfg["image_size"]),
                norm_mean=[float(x) for x in m_cfg["normalize_mean"]],
                norm_std=[float(x) for x in m_cfg["normalize_std"]],
            )
            emb_dict[bb_name] = mat
            norms = np.linalg.norm(mat, axis=1)
            deep_summary_rows.append({
                "track": track_name,
                "modality": modality,
                "backbone": bb_name,
                "embedding_dim": meta["embedding_dim"],
                "n_samples": meta["n_samples"],
                "checkpoint_loaded": meta["checkpoint_loaded"],
                "checkpoint_path": meta["checkpoint_path"],
                "l2_norm_mean": float(np.mean(norms)),
                "l2_norm_std": float(np.std(norms)),
            })

        np.savez_compressed(
            out_root / f"{track_name}_deep_embeddings.npz",
            sample_id=df_trk["sample_id"].to_numpy(),
            label=df_trk[lbl_col].to_numpy(),
            split=df_trk["split"].to_numpy(),
            **emb_dict,
        )

        # 3. Exploratory PCA / t-SNE / UMAP across Handcrafted + 3 Deep Backbones
        dr_frames: list[pd.DataFrame] = []
        spaces = [("handcrafted", X_hc_mat)] + [(bb, emb_dict[bb]) for bb in ("resnet50", "densenet121", "efficientnet_b0")]
        for space_name, X_mat in spaces:
            x_pca, x_tsne, x_umap, cum_var = _run_dim_reduction(X_mat, seed=42)
            for k_idx, cv in enumerate(cum_var, start=1):
                pca_var_rows.append({"track": track_name, "feature_space": space_name, "n_components": k_idx, "cum_explained_variance": float(cv)})
            dr_frames.append(pd.DataFrame({
                "track": track_name,
                "feature_space": space_name,
                "sample_id": df_trk["sample_id"].to_numpy(),
                "label": df_trk[lbl_col].to_numpy(),
                "split": df_trk["split"].to_numpy(),
                "pca_1": x_pca[:, 0],
                "pca_2": x_pca[:, 1],
                "tsne_1": x_tsne[:, 0],
                "tsne_2": x_tsne[:, 1],
                "umap_1": x_umap[:, 0],
                "umap_2": x_umap[:, 1],
            }))
        dr_df = pd.concat(dr_frames, ignore_index=True)
        dr_df.to_csv(out_root / f"dim_reduction_{track_name}.csv", index=False)
        dr_tables[track_name] = dr_df

    deep_summary_df = pd.DataFrame(deep_summary_rows)
    deep_summary_df.to_csv(out_root / "deep_extraction_summary.csv", index=False)
    pca_var_df = pd.DataFrame(pca_var_rows)
    pca_var_df.to_csv(out_root / "pca_variance_explained.csv", index=False)

    # 4. Feature Correlation Matrices (interpretable non-histogram subset for clear heatmaps)
    f_corr_cols = [c for c in f_feat_cols if "_hist_" not in c and "_bin_" not in c]
    o_corr_cols = [c for c in o_feat_cols if "_hist_" not in c and "_bin_" not in c]
    f_corr = f_hc_df[f_corr_cols].corr()
    o_corr = o_hc_df[o_corr_cols].corr()
    f_corr.to_csv(out_root / "fundus_handcrafted_correlation.csv")
    o_corr.to_csv(out_root / "oct_handcrafted_correlation.csv")

    # 5. Leak-Free Handcrafted ML Baseline & Feature Importance
    clf_all: list[dict[str, Any]] = []
    imp_tables: dict[str, pd.DataFrame] = {}
    for track_name, df_sub, feat_cols_trk, lbl_col in (
        ("fundus_3class", f_hc_df, f_feat_cols, "label"),
        ("oct_individual", o_hc_df, o_feat_cols, "label"),
        ("oct_severity", osv_hc_df, o_feat_cols, "label"),
    ):
        X_sub = df_sub[feat_cols_trk].to_numpy(dtype=np.float32)
        c_rows, imp_df = _evaluate_handcrafted_ml(df_sub, X_sub, feat_cols_trk, track_name, lbl_col)
        clf_all.extend(c_rows)
        imp_df.to_csv(out_root / f"feature_importance_{track_name}.csv", index=False)
        imp_tables[track_name] = imp_df

    clf_df = pd.DataFrame(clf_all)
    clf_df.to_csv(out_root / "handcrafted_ml_baseline_results.csv", index=False)

    # 6. Generate Publication Figures (PNG + PDF)
    fig_paths = generate_all_feature_figures(
        f_hc_df=f_hc_df,
        o_hc_df=o_hc_df,
        f_corr=f_corr,
        o_corr=o_corr,
        pca_var_df=pca_var_df,
        dr_tables=dr_tables,
        imp_tables=imp_tables,
        clf_df=clf_df,
        fig_dir=fig_dir,
    )

    manifest = {
        "output_dir": str(out_root),
        "figures_dir": str(fig_dir),
        "fundus_handcrafted_dim": len(f_feat_cols),
        "fundus_hog_dim": int(f_hog.shape[1]),
        "oct_handcrafted_dim": len(o_feat_cols),
        "oct_hog_dim": int(o_hog.shape[1]),
        "deep_embeddings": deep_summary_rows,
        "handcrafted_ml_baseline": clf_all,
        "figures_generated": [str(p) for p in fig_paths],
    }
    (out_root / "feature_extraction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

