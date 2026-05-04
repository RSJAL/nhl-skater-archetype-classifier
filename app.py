"""
app.py — NHL Skater Archetype Explorer
Streamlit dashboard: pick a player, see their archetype + nearest neighbours.

Run with:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).parent

CLUSTER_FEATURES = [
    "G_60", "A_60", "SOG_60", "BLK_60", "HIT_60",
    "PP_PTS_60", "ES_TOI_GP", "PP_TOI_GP", "SH_TOI_GP",
    "PIM_GP", "Fights",
    "xGF_pct_5v5", "ixG_60_5v5",
]

RADAR_STATS = [
    "G_60", "A_60", "SOG_60", "xGF_pct_5v5", "BLK_60",
    "HIT_60", "Fights", "SH_TOI_GP", "PP_TOI_GP", "ES_TOI_GP",
]
RADAR_LABELS = [
    "G/60", "A/60", "SOG/60", "xGF%", "Blocks/60",
    "Hits/60", "Fights", "SH TOI/GP", "PP TOI/GP", "ES TOI/GP",
]

# Stats where 0 is not a meaningful floor — use 5th percentile instead
RANGE_STATS = {"xGF_pct_5v5", "ES_TOI_GP", "PP_TOI_GP", "SH_TOI_GP"}

CARD_STATS = [
    ("GP",          "GP",          None,  ".0f"),
    ("TOI/GP",      "TOI_GP",      None,  ".1f"),
    ("G/60",        "G_60",        1.5,   ".2f"),
    ("A/60",        "A_60",        2.0,   ".2f"),
    ("SOG/60",      "SOG_60",      12.0,  ".2f"),
    ("xGF% (5v5)",  "xGF_pct_5v5", 65.0,  ".1f"),
    ("Hits/60",     "HIT_60",      15.0,  ".2f"),
    ("Blocks/60",   "BLK_60",      8.0,   ".2f"),
]

ARCHETYPE_COLORS = {
    # Forwards
    "Defensive Forward":    "#81C784",
    "Producer":             "#CE93D8",
    "Enforcer":             "#F06292",
    "Playmaker":            "#4FC3F7",
    # Defensemen
    "Defensive Defenseman": "#90A4AE",
    "Playmaker":            "#FFB74D",
    "Two-Way Player":       "#4DB6AC",
    # "Enforcer" shared with forwards — same color
}

# =============================================================================
# Data loading
# =============================================================================

@st.cache_data
def load_data():
    df = pd.read_csv(HERE / "players_clustered.csv")

    scaler_f = joblib.load(HERE / "scaler_f.pkl")
    scaler_d = joblib.load(HERE / "scaler_d.pkl")

    df_f = df[df["Group"] == "F"].copy().reset_index(drop=True)
    df_d = df[df["Group"] == "D"].copy().reset_index(drop=True)

    X_f = scaler_f.transform(df_f[CLUSTER_FEATURES].values)
    X_d = scaler_d.transform(df_d[CLUSTER_FEATURES].values)

    # Radar ceilings (95th pct) and floors (5th pct for range-stats, else 0)
    radar_max   = {s: float(np.percentile(df[s].dropna(), 95)) for s in RADAR_STATS}
    radar_floor = {s: float(np.percentile(df[s].dropna(), 5)) if s in RANGE_STATS else 0.0
                   for s in RADAR_STATS}

    return df_f, df_d, X_f, X_d, radar_max, radar_floor


@st.cache_resource
def build_nn(_X):
    nn = NearestNeighbors(metric="euclidean")
    nn.fit(_X)
    return nn

# =============================================================================
# Chart helpers
# =============================================================================

def make_radar(rows, selected_name, radar_max, radar_floor):
    labels_closed = RADAR_LABELS + [RADAR_LABELS[0]]
    fig = go.Figure()

    for row in rows:
        is_selected = row["Name"] == selected_name
        color = ARCHETYPE_COLORS.get(row["Auto_Label"], "#888888")

        raw  = [float(row[s]) for s in RADAR_STATS]
        norm = [
            np.clip((v - radar_floor[s]) / (radar_max[s] - radar_floor[s]), 0.0, 1.0)
            for v, s in zip(raw, RADAR_STATS)
        ]
        norm_closed = norm + [norm[0]]

        hover = (
            f"<b>{row['Name']}</b><br>"
            + "<br>".join(f"{lbl}: {raw[i]:.2f}" for i, lbl in enumerate(RADAR_LABELS))
            + f"<br>Archetype: {row['Auto_Label']}"
            + "<extra></extra>"
        )

        fig.add_trace(go.Scatterpolar(
            r=norm_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor=color,
            opacity=0.75 if is_selected else 0.18,
            line=dict(
                color=color,
                width=3.5 if is_selected else 1.5,
                dash="solid" if is_selected else "dot",
            ),
            name=row["Name"],
            hovertemplate=hover,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#f7f7f7",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["25%", "50%", "75%", "95th"],
                tickfont=dict(size=8, color="#aaa"),
                gridcolor="#ddd",
                linecolor="#ddd",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#333"),
                gridcolor="#ddd",
                linecolor="#ddd",
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.32,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=70, r=70, t=20, b=110),
        height=540,
        paper_bgcolor="white",
    )
    return fig


def stat_bar_html(label, value, max_val, color, fmt):
    disp = f"{value:{fmt}}"
    if max_val is None:
        return (
            f'<div style="margin:4px 0;font-size:13px;color:#333">'
            f'<span style="color:#888;font-size:12px">{label}:</span>&nbsp;<b>{disp}</b></div>'
        )
    pct = min(float(value) / max_val * 100, 100)
    return (
        f'<div style="margin:4px 0;display:flex;align-items:center;gap:6px">'
        f'  <span style="width:92px;font-size:12px;color:#555">{label}</span>'
        f'  <span style="width:40px;font-size:12px;font-weight:700;text-align:right">{disp}</span>'
        f'  <span style="flex:1;background:#eee;border-radius:4px;height:11px;display:inline-block">'
        f'    <span style="display:block;background:{color};border-radius:4px;'
        f'           width:{pct}%;height:11px"></span>'
        f'  </span>'
        f'</div>'
    )

# =============================================================================
# App
# =============================================================================

st.set_page_config(
    page_title="NHL Skater Archetype Explorer",
    page_icon="🏒",
    layout="wide",
)

df_f, df_d, X_f, X_d, radar_max, radar_floor = load_data()
nn_f = build_nn(X_f)
nn_d = build_nn(X_d)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🏒 NHL Skater Archetypes")
    st.caption(
        "K-Means clustering on 14 features (2022–23 to 2024–25 regular seasons). "
        "Forwards and defensemen clustered separately. "
        "Similarity: Euclidean distance in scaled feature space."
    )
    st.divider()

    group      = st.radio("Position group", ["Forwards", "Defensemen"])
    is_forward = group == "Forwards"
    df_group   = df_f if is_forward else df_d
    X_group    = X_f  if is_forward else X_d
    nn_model   = nn_f if is_forward else nn_d

    default    = "Connor McDavid" if is_forward else "Cale Makar"
    names      = sorted(df_group["Name"].tolist())
    default_idx = names.index(default) if default in names else 0

    selected_name  = st.selectbox("Choose a player", names, index=default_idx)
    n_neighbours   = st.slider("Neighbours to show", 3, 10, 5)
    same_archetype = st.checkbox("Same-archetype neighbours only", value=False)

    st.divider()
    st.markdown("**Archetypes**")
    for archetype in sorted(df_group["Auto_Label"].unique()):
        color = ARCHETYPE_COLORS.get(archetype, "#888")
        st.markdown(
            f'<span style="display:inline-block;width:12px;height:12px;background:{color};'
            f'border-radius:50%;margin-right:6px;vertical-align:middle"></span>'
            f'<span style="font-size:13px">{archetype}</span>',
            unsafe_allow_html=True,
        )

# ── Look up selected player ───────────────────────────────────────────────────
idx      = df_group[df_group["Name"] == selected_name].index[0]
selected = df_group.loc[idx]
x_sel    = X_group[idx].reshape(1, -1)

distances, indices = nn_model.kneighbors(x_sel, n_neighbors=n_neighbours + 25)
distances, indices = distances[0], indices[0]

mask      = indices != idx
distances = distances[mask]
indices   = indices[mask]

neighbours_df = df_group.iloc[indices].copy()
neighbours_df["Distance"] = np.round(distances, 3)

if same_archetype:
    neighbours_df = neighbours_df[neighbours_df["Auto_Label"] == selected["Auto_Label"]]

neighbours_df = neighbours_df.head(n_neighbours).reset_index(drop=True)

# ── Main layout ───────────────────────────────────────────────────────────────
col_card, col_radar = st.columns([1, 2], gap="large")

with col_card:
    archetype = selected["Auto_Label"]
    color     = ARCHETYPE_COLORS.get(archetype, "#888")

    st.markdown(f"## {selected_name}")
    st.markdown(
        f"**{selected.get('Pos', '')}** &nbsp;·&nbsp; {selected.get('Nationality', '')}",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="background:{color}28;border-left:4px solid {color};'
        f'padding:8px 14px;border-radius:6px;font-weight:700;font-size:15px;margin:8px 0 16px">'
        f'{archetype}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Stats**")
    for label, col_name, max_val, fmt in CARD_STATS:
        if col_name in selected.index and pd.notna(selected[col_name]):
            st.markdown(
                stat_bar_html(label, selected[col_name], max_val, color, fmt),
                unsafe_allow_html=True,
            )

with col_radar:
    st.markdown(f"#### {selected_name} vs nearest neighbours")
    rows_to_plot = [selected] + [neighbours_df.iloc[i] for i in range(len(neighbours_df))]
    st.plotly_chart(make_radar(rows_to_plot, selected_name, radar_max, radar_floor), use_container_width=True)

# ── Neighbour table ───────────────────────────────────────────────────────────
st.divider()
st.markdown(f"#### {n_neighbours} Nearest Neighbours")

display_cols = [
    "Name", "Pos", "Auto_Label", "GP",
    "G_60", "A_60", "SOG_60", "xGF_pct_5v5",
    "HIT_60", "BLK_60", "Distance",
]
display_cols = [c for c in display_cols if c in neighbours_df.columns]
display_df = neighbours_df[display_cols].rename(columns={"Auto_Label": "Archetype"})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Distance":    st.column_config.NumberColumn(format="%.3f"),
        "GP":          st.column_config.NumberColumn(format="%d"),
        "G_60":        st.column_config.ProgressColumn("G/60",       min_value=0,  max_value=2.0,  format="%.2f"),
        "A_60":        st.column_config.ProgressColumn("A/60",       min_value=0,  max_value=2.5,  format="%.2f"),
        "SOG_60":      st.column_config.ProgressColumn("SOG/60",     min_value=0,  max_value=12.0, format="%.2f"),
        "xGF_pct_5v5": st.column_config.ProgressColumn("xGF% (5v5)", min_value=40, max_value=65,   format="%.1f"),
        "HIT_60":      st.column_config.ProgressColumn("Hits/60",    min_value=0,  max_value=15.0, format="%.2f"),
        "BLK_60":      st.column_config.ProgressColumn("Blocks/60",  min_value=0,  max_value=8.0,  format="%.2f"),
    },
)
