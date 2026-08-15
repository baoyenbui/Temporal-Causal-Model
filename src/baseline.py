import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    from lingam import DirectLiNGAM
    LINGAM_AVAILABLE = True
except ImportError:
    LINGAM_AVAILABLE = False


NODE_FEATURES = [
    'engagement_proxy_scaled',
    'time_delta',
    'step_index_scaled',
    'question_difficulty',
    'student_cluster',
    'content_group'
]

ACTIONABLE_FEATURES = {
    'engagement_proxy_scaled',
    'time_delta',
    'step_index_scaled',
    'question_difficulty'
}
TARGET_COL = 'IsCorrect'


def build_nontemporal_features(clean_main_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_main_df.copy()
    df = df[df[TARGET_COL].isin([0, 1])].copy()

    content_col = None
    for candidate in ('ParentId_encoded', 'SubjectId_encoded'):
        if candidate in df.columns:
            content_col = candidate
            break
    if content_col is None:
        raise ValueError("No content-group column found (expected 'ParentId_encoded' or 'SubjectId_encoded' "
                         "from the preprocessing step's categorical encoding).")
    df['content_group'] = df[content_col]

    required = NODE_FEATURES + ['UserId', TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for the non-temporal baseline: {missing}. "
                         f"Check that the preprocessing notebook produced these before calling this function.")

    out = df[required].dropna().reset_index(drop=True)
    print(f"Non-temporal baseline features built: {len(out)} rows, {out['UserId'].nunique()} students, "
          f"class balance IsCorrect=1: {out[TARGET_COL].mean():.1%}")
    return out


def split_by_user(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    user_ids = df['UserId'].unique()
    rng.shuffle(user_ids)
    n = len(user_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_users = set(user_ids[:n_train])
    val_users = set(user_ids[n_train:n_train + n_val])
    test_users = set(user_ids[n_train + n_val:])

    train_df = df[df['UserId'].isin(train_users)].reset_index(drop=True)
    val_df = df[df['UserId'].isin(val_users)].reset_index(drop=True)
    test_df = df[df['UserId'].isin(test_users)].reset_index(drop=True)

    print(f"Split by UserId: train={len(train_df)} rows ({len(train_users)} students), "
          f"val={len(val_df)} rows ({len(val_users)} students), "
          f"test={len(test_df)} rows ({len(test_users)} students)")
    return train_df, val_df, test_df


def bootstrap_directlingam(
    train_df: pd.DataFrame,
    n_bootstrap: int = 20,
    stability_threshold: float = 0.6,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not LINGAM_AVAILABLE:
        raise ImportError("The 'lingam' package is required for this baseline: "
                          "pip install lingam --break-system-packages")

    X = train_df[NODE_FEATURES].values.astype(float)
    rng = np.random.default_rng(seed)
    n = len(X)

    edge_counts: Dict[Tuple[str, str], int] = {}
    edge_strengths: Dict[Tuple[str, str], list] = {}
    successful = 0

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        X_boot = X[idx]
        try:
            model = DirectLiNGAM()
            model.fit(X_boot)
        except Exception as e:
            print(f"Bootstrap iteration {i + 1}/{n_bootstrap}: DirectLiNGAM failed ({type(e).__name__}: {e}), skipping")
            continue
        successful += 1

        adj = model.adjacency_matrix_
        for i_row in range(adj.shape[0]):
            for j_col in range(adj.shape[1]):
                if i_row == j_col:
                    continue
                w = adj[i_row, j_col]
                if abs(w) > 1e-6:
                    key = (NODE_FEATURES[j_col], NODE_FEATURES[i_row])
                    edge_counts[key] = edge_counts.get(key, 0) + 1
                    edge_strengths.setdefault(key, []).append(w)

    print(f"DirectLiNGAM bootstrap: {successful}/{n_bootstrap} successful runs")

    rows = []
    for key, count in edge_counts.items():
        source, target = key
        rows.append({
            'source': source,
            'target': target,
            'stability': count / successful if successful else 0.0,
            'mean_strength': float(np.mean(edge_strengths[key])),
            'std_strength': float(np.std(edge_strengths[key])),
            'n_occurrences': count
        })

    edges_df = pd.DataFrame(rows)
    if len(edges_df) > 0:
        edges_df = edges_df.sort_values('stability', ascending=False).reset_index(drop=True)
        stable_edges_df = edges_df[edges_df['stability'] >= stability_threshold].reset_index(drop=True)
    else:
        stable_edges_df = edges_df

    print(f"Edges (pooled): {len(edges_df)}, stable (>= {stability_threshold:.0%}): {len(stable_edges_df)}")
    return edges_df, stable_edges_df


def identify_actionable_upstream(
    stable_edges_df: pd.DataFrame,
    actionable_features: set = ACTIONABLE_FEATURES
) -> List[str]:
    G = nx.DiGraph()
    for f in NODE_FEATURES:
        G.add_node(f)
    for _, row in stable_edges_df.iterrows():
        G.add_edge(row['source'], row['target'])

    sinks = {n for n in G.nodes if G.out_degree(n) == 0}

    eligible = sorted(f for f in actionable_features if f in G.nodes and f not in sinks)
    if not eligible:
        eligible = sorted(f for f in actionable_features if f in G.nodes)

    excluded = sorted(set(NODE_FEATURES) - set(eligible))
    print(f"Actionable + causally-upstream features: {eligible}")
    print(f"Excluded (non-actionable or pure-sink under the stable graph): {excluded}")
    return eligible


def train_success_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str] = NODE_FEATURES,
    seed: int = 42
):
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

    best_model, best_f1, best_n = None, -1.0, None
    for n_estimators in (100, 200, 300):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        f1 = f1_score(y_val, val_pred)
        if f1 > best_f1:
            best_f1, best_model, best_n = f1, model, n_estimators

    print(f"Model selection: best n_estimators={best_n} (val F1={best_f1:.3f})")

    test_pred = best_model.predict(X_test)
    test_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': float(accuracy_score(y_test, test_pred)),
        'f1_weighted': float(f1_score(y_test, test_pred, average='weighted')),
        'auc': float(roc_auc_score(y_test, test_proba)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, test_proba)))
    }
    print(f"Test performance: accuracy={metrics['accuracy']:.3f}, f1_weighted={metrics['f1_weighted']:.3f}, "
          f"auc={metrics['auc']:.3f}, rmse={metrics['rmse']:.3f}")
    return best_model, metrics


def generate_baseline_counterfactuals(
    model: RandomForestClassifier,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    actionable_cols: List[str],
    feature_cols: List[str] = NODE_FEATURES,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    n_instances: int = 30,
    lambda_causal: float = 0.5,
    n_stability_trials: int = 20,
    noise_frac: float = 0.05,
    seed: int = 42
) -> pd.DataFrame:
    if not actionable_cols:
        print("No actionable+upstream features available under the stable graph; no counterfactuals can be generated.")
        return pd.DataFrame()

    bounds = {c: (float(train_df[c].quantile(lower_q)), float(train_df[c].quantile(upper_q))) for c in feature_cols}
    scaler = StandardScaler().fit(train_df[feature_cols])
    actionable_idx = [feature_cols.index(c) for c in actionable_cols]

    rng = np.random.default_rng(seed)
    test_sample = test_df.sample(n=min(n_instances, len(test_df)), random_state=seed)

    results = []
    for _, instance in test_sample.iterrows():
        x = instance[feature_cols].values.astype(float)
        pred = int(model.predict(x.reshape(1, -1))[0])
        opposite_class = 1 - pred
        pool = train_df[train_df[TARGET_COL] == opposite_class]
        if len(pool) == 0:
            continue

        pool_scaled_full = scaler.transform(pool[feature_cols].values)
        x_scaled_full = scaler.transform(x.reshape(1, -1))[0]

        pool_actionable_scaled = pool_scaled_full[:, actionable_idx]
        x_actionable_scaled = x_scaled_full[actionable_idx]
        pool_actionable_raw = pool[actionable_cols].values.astype(float)

        dists = np.linalg.norm(pool_actionable_scaled - x_actionable_scaled, axis=1)
        n_changed = (np.abs(pool_actionable_scaled - x_actionable_scaled) > 0.1).sum(axis=1)
        score = dists + lambda_causal * n_changed

        best_idx = int(np.argmin(score))
        candidate_actionable_raw = pool_actionable_raw[best_idx]

        x_cf = x.copy()
        for i, c in enumerate(actionable_cols):
            lo, hi = bounds[c]
            idx = feature_cols.index(c)
            x_cf[idx] = float(np.clip(candidate_actionable_raw[i], lo, hi))

        pred_cf = int(model.predict(x_cf.reshape(1, -1))[0])
        proba_orig = float(model.predict_proba(x.reshape(1, -1))[0][1])
        proba_cf = float(model.predict_proba(x_cf.reshape(1, -1))[0][1])

        x_cf_scaled = scaler.transform(x_cf.reshape(1, -1))[0]
        proximity = float(np.linalg.norm(x_scaled_full[actionable_idx] - x_cf_scaled[actionable_idx]))
        sparsity = int((np.abs(x_cf - x) > 1e-6).sum())

        stable_flips = 0
        for _ in range(n_stability_trials):
            x_cf_noisy = x_cf.copy()
            for idx in actionable_idx:
                span = bounds[feature_cols[idx]][1] - bounds[feature_cols[idx]][0]
                x_cf_noisy[idx] += rng.normal(0, noise_frac * (span if span > 0 else 1.0))
                lo, hi = bounds[feature_cols[idx]]
                x_cf_noisy[idx] = np.clip(x_cf_noisy[idx], lo, hi)
            pred_noisy = int(model.predict(x_cf_noisy.reshape(1, -1))[0])
            if pred_noisy == pred_cf:
                stable_flips += 1
        perturbation_stability = stable_flips / n_stability_trials

        results.append({
            'original_pred': pred,
            'cf_pred': pred_cf,
            'flip_success': bool(pred_cf != pred),
            'proba_original': proba_orig,
            'proba_cf': proba_cf,
            'proba_change': proba_cf - proba_orig,
            'proximity': proximity,
            'sparsity': sparsity,
            'perturbation_stability': perturbation_stability
        })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        print(f"Baseline CF generated for {len(results_df)} test instances: "
              f"flip_success_rate={results_df['flip_success'].mean():.1%}, "
              f"mean_proba_change={results_df['proba_change'].mean():+.3f}, "
              f"mean_proximity={results_df['proximity'].mean():.3f}, "
              f"mean_sparsity={results_df['sparsity'].mean():.2f}, "
              f"mean_perturbation_stability={results_df['perturbation_stability'].mean():.3f}")
    return results_df


def run_nontemporal_baseline(
    clean_main_df: pd.DataFrame,
    n_bootstrap: int = 20,
    stability_threshold: float = 0.6,
    n_cf_instances: int = 30,
    seed: int = 42
) -> Dict:
    print("=" * 60)
    print("NON-TEMPORAL BASELINE (DirectLiNGAM + bootstrap + RF + opposite-class CF)")
    print("=" * 60)

    df = build_nontemporal_features(clean_main_df)
    train_df, val_df, test_df = split_by_user(df, seed=seed)

    edges_df, stable_edges_df = bootstrap_directlingam(
        train_df, n_bootstrap=n_bootstrap, stability_threshold=stability_threshold, seed=seed
    )
    if len(stable_edges_df) > 0:
        print("\nStable edges:")
        print(stable_edges_df[['source', 'target', 'stability', 'mean_strength']].to_string(index=False))

    actionable_upstream = identify_actionable_upstream(stable_edges_df)

    model, clf_metrics = train_success_classifier(train_df, val_df, test_df, seed=seed)

    cf_df = generate_baseline_counterfactuals(
        model, train_df, test_df, actionable_cols=actionable_upstream,
        n_instances=n_cf_instances, seed=seed
    )

    return {
        'features_df': df,
        'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
        'edges_df': edges_df, 'stable_edges_df': stable_edges_df,
        'actionable_upstream': actionable_upstream,
        'model': model, 'classifier_metrics': clf_metrics,
        'cf_results': cf_df
    }