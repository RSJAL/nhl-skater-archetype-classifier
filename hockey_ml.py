"""
hockey_ml.py
------------
Clustering pipeline for NHL skaters (2022-23 to 2025-26).
Forwards and defensemen are clustered separately.

Workflow:
  1. Load players_raw.csv, split into Forwards / Defensemen
  2. Scale features within each group
  3. K-Means selection plots (K=2-15) per group
  4. Fit at HIGH_K_F / HIGH_K_D, auto-label clusters
  5. Radar charts per group
  6. Export players_clustered.csv (combined, with Group column)
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
import joblib

SEED = 42
np.random.seed(SEED)

# ── Tune these after inspecting the selection plots ───────────────────────────
HIGH_K_F = 4   # K for forwards
HIGH_K_D = 4   # K for defensemen
# ─────────────────────────────────────────────────────────────────────────────

# Manual label overrides — applied after auto-labeling (empty = use auto)
MANUAL_LABELS_F: dict = {
    0: "Defensive Forward",
    1: "Producer",
    2: "Enforcer",
    3: "Playmaker",
}
MANUAL_LABELS_D: dict = {
    0: "Defensive Defenseman",
    1: "Playmaker",
    2: "Two-Way Player",
    3: "Enforcer",
}

HERE = Path(__file__).parent

CLUSTER_FEATURES = [
    "G_60", "A_60", "SOG_60", "BLK_60", "HIT_60",
    "PP_PTS_60", "ES_TOI_GP", "PP_TOI_GP", "SH_TOI_GP",
    "PIM_GP", "Fights",
    "xGF_pct_5v5", "ixG_60_5v5",
]

RADAR_STATS  = ["G_60", "A_60", "SOG_60", "xGF_pct_5v5", "BLK_60", "HIT_60", "Fights", "SH_TOI_GP", "PP_TOI_GP", "ES_TOI_GP"]
RADAR_LABELS = ["G/60", "A/60", "SOG/60", "xGF%", "Blocks/60", "Hits/60", "Fights", "SH TOI/GP", "PP TOI/GP", "ES TOI/GP"]

PALETTE = [
    "#4FC3F7", "#81C784", "#CE93D8", "#FFB74D", "#F06292",
    "#4DB6AC", "#FF8A65", "#90A4AE", "#FFF176", "#A5D6A7",
    "#80CBC4", "#EF9A9A",
]

STAT_LABELS_F = {
    "G_60":         "Goal Scorer",
    "A_60":         "Playmaker",
    "SOG_60":       "Shot Generator",
    "BLK_60":       "Defensive Forward",
    "HIT_60":       "Power Forward",
    "PP_PTS_60":    "PP Producer",
    "PP_TOI_GP":    "PP Specialist",
    "SH_TOI_GP":    "PK Specialist",
    "PIM_GP":       "Agitator",
    "Fights":       "Enforcer",
    "ES_TOI_GP":    "ES Workhorse",
    "CF_pct_5v5":   "Possession Driver",
    "xGF_pct_5v5":  "Shot Quality",
    "ixG_60_5v5":   "Shot Generator",
}

STAT_LABELS_D = {
    "G_60":         "Offensive D",
    "A_60":         "Offensive D",
    "SOG_60":       "Offensive D",
    "BLK_60":       "Shutdown D",
    "HIT_60":       "Physical D",
    "PP_PTS_60":    "PP Quarterback",
    "PP_TOI_GP":    "PP Quarterback",
    "SH_TOI_GP":    "PK Specialist",
    "PIM_GP":       "Agitator D",
    "Fights":       "Enforcer D",
    "ES_TOI_GP":    "Two-Way D",
    "CF_pct_5v5":   "Two-Way D",
    "xGF_pct_5v5":  "Offensive D",
    "ixG_60_5v5":   "Offensive D",
}

# =============================================================================
# 1. Load and split
# =============================================================================
df = pd.read_csv(HERE / "players_raw.csv")
print(f"Loaded {len(df)} qualified players")

before = len(df)
df = df.dropna(subset=CLUSTER_FEATURES).reset_index(drop=True)
if len(df) < before:
    print(f"  Dropped {before - len(df)} rows with missing cluster features")

# Radar ceilings/floors computed globally so F and D are on the same scale
# Range-stats use 5th pct floor (their variation lives in a narrow band above 0)
RANGE_STATS  = {"xGF_pct_5v5", "ES_TOI_GP", "PP_TOI_GP", "SH_TOI_GP"}
RADAR_MAX:   dict = {}
RADAR_FLOOR: dict = {}
for stat in RADAR_STATS:
    RADAR_MAX[stat]   = float(np.percentile(df[stat].dropna(), 95))
    RADAR_FLOOR[stat] = float(np.percentile(df[stat].dropna(), 5)) if stat in RANGE_STATS else 0.0
print(f"\nRadar ceilings (95th pct, global): { {k: round(v,2) for k,v in RADAR_MAX.items()} }")
print(f"Radar floors   (5th pct, range stats): { {k: round(v,2) for k,v in RADAR_FLOOR.items() if k in RANGE_STATS} }")

df_f = df[df["Pos"].isin(["C", "LW", "RW"])].copy().reset_index(drop=True)
df_d = df[df["Pos"] == "D"].copy().reset_index(drop=True)
print(f"\nForwards: {len(df_f)}   Defensemen: {len(df_d)}")

# =============================================================================
# Helper functions
# =============================================================================

def darken(hex_color, factor=0.6):
    r, g, b = mcolors.to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)

def tint(hex_color, factor=0.12):
    r, g, b = mcolors.to_rgb(hex_color)
    return (r+(1-r)*(1-factor), g+(1-g)*(1-factor), b+(1-b)*(1-factor))


def draw_radar(ax, values, labels, color, title, n_players):
    N      = len(labels)
    angles = [math.pi/2 - n/N * 2*math.pi for n in range(N)]
    vx     = np.array([math.cos(a) for a in angles])
    vy     = np.array([math.sin(a) for a in angles])

    ax.set_facecolor(tint(color, 0.12))
    ax.add_patch(MplPolygon(list(zip(vx, vy)), closed=True,
                            facecolor=color, alpha=0.18, edgecolor="none", zorder=0))
    for level in [0.25, 0.5, 0.75]:
        ax.add_patch(MplPolygon(list(zip(vx*level, vy*level)), closed=True,
                                facecolor="none", edgecolor="#cccccc", lw=0.8, zorder=1))
    ax.add_patch(MplPolygon(list(zip(vx, vy)), closed=True,
                            facecolor="none", edgecolor=color, lw=3.5, zorder=5))
    for x, y in zip(vx, vy):
        ax.plot([0, x], [0, y], color="#cccccc", lw=0.8, zorder=2)

    scaled = np.array([
        np.clip((v - RADAR_FLOOR[s]) / (RADAR_MAX[s] - RADAR_FLOOR[s]), 0.0, 1.0)
        for v, s in zip(values, RADAR_STATS)
    ])
    px = np.append(vx*scaled, vx[0]*scaled[0])
    py = np.append(vy*scaled, vy[0]*scaled[0])
    ax.fill(px, py, color=color, alpha=0.35, zorder=3)
    ax.plot(px, py, color=color, lw=2.5, zorder=4)
    ax.scatter(vx*scaled, vy*scaled, color=color, s=30, zorder=6,
               edgecolors="white", linewidths=0.8)

    for i, val in enumerate(values):
        nudge = 0.15
        ax.text(vx[i]*scaled[i] + vx[i]*nudge,
                vy[i]*scaled[i] + vy[i]*nudge,
                f"{val:.2f}", fontsize=6, color="#555",
                fontweight="bold", ha="center", va="center", zorder=7)

    for i, label in enumerate(labels):
        ax.text(vx[i]*1.32, vy[i]*1.32, label,
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="#222", zorder=6)

    ax.set_xlim(-1.65, 1.65); ax.set_ylim(-1.65, 1.65)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, size=9, pad=10, color="#222", fontweight="bold")
    ax.text(0, -1.6, f"n={n_players}", ha="center", va="top",
            fontsize=7, color=darken(color, 0.65), zorder=6)


def auto_label(means_df, g_mean, g_std, stat_labels, bottom_label):
    """Assign a descriptive label to each cluster based on its standout stat."""
    total  = means_df.sum(axis=1)
    min_c  = total.idxmin()
    labels, seen = {}, {}
    for c, row in means_df.iterrows():
        z    = (row - g_mean) / g_std
        best = z.idxmax()
        if c == min_c and z.max() < 0.35:
            base = bottom_label
        elif z.max() < 0.35:
            base = "Two-Way"
        else:
            base = stat_labels.get(best, best)
        if base in seen:
            seen[base] += 1
            labels[c] = f"{base} {seen[base]}"
        else:
            seen[base] = 1
            labels[c]  = base
    return labels


def run_pipeline(df_group, group_name, high_k, stat_labels, bottom_label, file_prefix, manual_labels=None):
    """
    Full K-Means pipeline for one position group.
    Returns df_group with Cluster and Auto_Label columns added.
    """
    print(f"\n{'='*70}")
    print(f"  {group_name.upper()}  (n={len(df_group)}, K={high_k})")
    print(f"{'='*70}")

    X_raw  = df_group[CLUSTER_FEATURES].values
    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)

    # ── K-Means selection K=2..15 ─────────────────────────────────────────────
    print(f"\nRunning K-Means selection K=2..15 ...")
    K_RANGE    = range(2, 16)
    inertias   = []
    sil_scores = []

    for k in K_RANGE:
        km     = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))
        print(f"  K={k:>2}  inertia={km.inertia_:.1f}  sil={sil_scores[-1]:.4f}")

    ks = list(K_RANGE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.plot(ks, inertias, "o-", color="#4FC3F7", lw=2, ms=6)
    ax.axvline(high_k, color="#F06292", lw=1.5, ls="--", label=f"K={high_k}")
    ax.set_xlabel("K", fontsize=12); ax.set_ylabel("Inertia", fontsize=12)
    ax.set_title(f"Elbow — {group_name}", fontsize=13, fontweight="bold")
    ax.set_xticks(ks); ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.plot(ks, sil_scores, "o-", color="#81C784", lw=2, ms=6)
    ax.axvline(high_k, color="#F06292", lw=1.5, ls="--", label=f"K={high_k}")
    ax.set_xlabel("K", fontsize=12); ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title(f"Silhouette — {group_name}", fontsize=13, fontweight="bold")
    ax.set_xticks(ks); ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    sel_path = HERE / f"kmeans_selection_{file_prefix}.png"
    plt.savefig(sel_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {sel_path.name}")

    # ── Fit at high_k ─────────────────────────────────────────────────────────
    print(f"\nFitting K={high_k} (n_init=50) ...")
    km_final = KMeans(n_clusters=high_k, random_state=SEED, n_init=50)
    df_group = df_group.copy()
    df_group["Cluster"] = km_final.fit_predict(X)

    cluster_means = df_group.groupby("Cluster")[CLUSTER_FEATURES].mean()
    g_mean        = df_group[CLUSTER_FEATURES].mean()
    g_std         = df_group[CLUSTER_FEATURES].std()

    cluster_labels = auto_label(cluster_means, g_mean, g_std, stat_labels, bottom_label)
    cluster_labels.update(manual_labels or {})   # apply any manual overrides
    df_group["Auto_Label"] = df_group["Cluster"].map(cluster_labels)

    sizes = df_group["Cluster"].value_counts().sort_index()
    print("Cluster sizes:", sizes.to_dict())
    print("Labels:", cluster_labels)

    print(f"\nCluster members (sample 6):")
    for c in range(high_k):
        members = df_group[df_group["Cluster"] == c]["Name"].tolist()
        line = f"  [{c:>2}] {cluster_labels[c]:<26} (n={len(members):>2}): {', '.join(members[:6])}"
        print(line.encode("ascii", "replace").decode("ascii"))

    # ── Radar charts ──────────────────────────────────────────────────────────
    NCOLS = min(high_k, 6)
    NROWS = math.ceil(high_k / NCOLS)
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 4, NROWS * 4.5))
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flatten()

    radar_means = df_group.groupby("Cluster")[RADAR_STATS].mean()

    for c in range(high_k):
        vals = radar_means.loc[c, RADAR_STATS].values
        draw_radar(axes_flat[c], vals, RADAR_LABELS,
                   PALETTE[c % len(PALETTE)], cluster_labels[c], sizes[c])

    for i in range(high_k, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(
        f"NHL {group_name} Archetypes — K={high_k} | Mean Stats (2022-23 to 2025-26)\n"
        f"Radar scale: global 95th percentile per stat",
        fontsize=13, color="#222", fontweight="bold", y=1.01
    )
    plt.tight_layout()
    radar_path = HERE / f"radar_{file_prefix}_K{high_k}.png"
    plt.savefig(radar_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {radar_path.name}")

    # ── Supervised evaluation — train/test split ──────────────────────────────
    y = df_group["Cluster"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"\nSupervised split: {len(X_train)} train / {len(X_test)} test")

    # ── KNN classifier (predict archetype for a new player) ───────────────────
    print("\n── KNN Classifier ──")
    knn_cv = {}
    for n in [3, 5, 7, 9]:
        cv = cross_val_score(KNeighborsClassifier(n_neighbors=n),
                             X_train, y_train, cv=5, scoring="accuracy")
        knn_cv[n] = cv.mean()
        print(f"  k={n}: CV {cv.mean():.3f} ± {cv.std():.3f}")

    best_k = max(knn_cv, key=knn_cv.get)
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train, y_train)
    print(f"  Best k={best_k} — test accuracy: {knn_final.score(X_test, y_test):.3f}")
    joblib.dump(knn_final, HERE / f"knn_model_{file_prefix}.pkl")
    print(f"  Saved knn_model_{file_prefix}.pkl")

    # ── Decision tree (cluster explainer) ─────────────────────────────────────
    print("\n── Decision Tree ──")
    dt_cv = {}
    for depth in [3, 4, 5, 6]:
        cv = cross_val_score(DecisionTreeClassifier(max_depth=depth, random_state=SEED),
                             X_train, y_train, cv=5, scoring="accuracy")
        dt_cv[depth] = cv.mean()
        print(f"  depth={depth}: CV {cv.mean():.3f} ± {cv.std():.3f}")

    best_depth = max(dt_cv, key=dt_cv.get)
    dt_final = DecisionTreeClassifier(max_depth=best_depth, random_state=SEED)
    dt_final.fit(X_train, y_train)
    print(f"  Best depth={best_depth} — test accuracy: {dt_final.score(X_test, y_test):.3f}")

    # Convert Z-score thresholds back to natural stat units for readability
    def raw_rules(tree, scaler, cluster_labels):
        tree_ = tree.tree_
        feat_names = [CLUSTER_FEATURES[i] if i != -2 else "leaf" for i in tree_.feature]
        lines = []
        def recurse(node, depth):
            indent = "|   " * depth
            if tree_.feature[node] != -2:
                fname = feat_names[node]
                fidx  = CLUSTER_FEATURES.index(fname)
                thresh = tree_.threshold[node] * scaler.scale_[fidx] + scaler.mean_[fidx]
                lines.append(f"{indent}|--- {fname} <= {thresh:.3f}")
                recurse(tree_.children_left[node],  depth + 1)
                lines.append(f"{indent}|--- {fname} >  {thresh:.3f}")
                recurse(tree_.children_right[node], depth + 1)
            else:
                pred = int(np.argmax(tree_.value[node][0]))
                lines.append(f"{indent}|--- class: {cluster_labels[pred]}")
        recurse(0, 0)
        return "\n".join(lines)

    print(f"\nDecision rules (depth={best_depth}, natural stat units):")
    print(raw_rules(dt_final, scaler, cluster_labels))

    fig, ax = plt.subplots(figsize=(22, 9))
    fig.patch.set_facecolor("white")
    plot_tree(dt_final, feature_names=CLUSTER_FEATURES,
              class_names=[cluster_labels[i] for i in range(high_k)],
              filled=True, rounded=True, fontsize=7, ax=ax)
    ax.set_title(f"NHL {group_name} Archetypes — Decision Tree (depth={best_depth})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    tree_path = HERE / f"decision_tree_{file_prefix}.png"
    plt.savefig(tree_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {tree_path.name}")

    # ── Save models ───────────────────────────────────────────────────────────
    joblib.dump(km_final, HERE / f"km_model_{file_prefix}.pkl")
    joblib.dump(scaler,   HERE / f"scaler_{file_prefix}.pkl")

    return df_group

# =============================================================================
# 2. Run pipeline for each group
# =============================================================================
df_f = run_pipeline(df_f, "Forwards",   HIGH_K_F, STAT_LABELS_F, "4th Liner",  "f", MANUAL_LABELS_F)
df_d = run_pipeline(df_d, "Defensemen", HIGH_K_D, STAT_LABELS_D, "Depth D",    "d", MANUAL_LABELS_D)

# =============================================================================
# 3. Combine and export
# =============================================================================
df_f["Group"] = "F"
df_d["Group"] = "D"
df_out = pd.concat([df_f, df_d], ignore_index=True)

out_cols = ["Name", "Pos", "Group", "Nationality", "GP",
            "TOI_GP", "ES_TOI_GP", "PP_TOI_GP", "SH_TOI_GP",
            *CLUSTER_FEATURES, "Height_in", "Weight_lbs", "BMI",
            "Cluster", "Auto_Label"]
out_cols = [c for c in out_cols if c in df_out.columns]

df_out[out_cols].sort_values(["Group", "Cluster", "Name"]).to_csv(
    HERE / "players_clustered.csv", index=False
)
print(f"\nSaved players_clustered.csv ({len(df_out)} players)")
print("\nDone. Inspect selection plots to tune HIGH_K_F / HIGH_K_D, then re-run.")
