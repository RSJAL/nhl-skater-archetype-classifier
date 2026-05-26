"""
similarity_explorer.py — NHL Skater Similarity Explorer
Three panels:
  1. Neighbour Web  — selected player in the centre, 6 nearest neighbours (combined F+D KNN)
  2. Comparison     — side-by-side dual stat bars
  3. PCA Scatter    — combined F+D universe, position-stratified archetype colours

Run with:  streamlit run similarity_explorer.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
STAT_FORMATS = {
    "G_60": ".2f", "A_60": ".2f", "SOG_60": ".2f",
    "xGF_pct_5v5": ".1f", "BLK_60": ".2f", "HIT_60": ".2f",
    "Fights": ".0f", "SH_TOI_GP": ".1f", "PP_TOI_GP": ".1f", "ES_TOI_GP": ".1f",
}

RANGE_STATS = {"xGF_pct_5v5", "ES_TOI_GP", "PP_TOI_GP", "SH_TOI_GP"}

# Position-stratified: F and D share label names but get distinct colours
ARCHETYPE_COLORS = {
    "F Defensive Forward":    "#81C784",
    "F Producer":             "#CE93D8",
    "F Enforcer":             "#F06292",
    "F Playmaker":            "#4FC3F7",
    "D Defensive Defenseman": "#90A4AE",
    "D Enforcer":             "#FF8A65",
    "D Playmaker":            "#FFB74D",
    "D Two-Way Player":       "#4DB6AC",
}

# Slot positions: index 0 = centre, 1–6 = neighbours
_POSITIONS = [
    (0.0,    0.132),
    (-1.584, 1.386),
    (0.0,    1.98),
    (1.584,  1.386),
    (-1.584, -1.122),
    (0.0,   -1.716),
    (1.584,  -1.122),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _last_name(full_name: str) -> str:
    parts = full_name.split()
    last  = parts[-1] if len(parts) > 1 else full_name
    return (last[:9] + "…") if len(last) > 10 else last


def _norm(val, stat, radar_max, radar_floor):
    floor = radar_floor[stat]
    ceil  = radar_max[stat]
    if ceil <= floor:
        return 0.0
    return float(np.clip((val - floor) / (ceil - floor), 0.0, 1.0))


def _strat_color(row) -> str:
    return ARCHETYPE_COLORS.get(row["Strat_Label"], "#888888")


# ── data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv(HERE / "players_clustered.csv")
    df = df.reset_index(drop=True)
    df["Strat_Label"] = df["Group"] + " " + df["Auto_Label"]

    # Combined scaler + feature matrix for cross-position KNN and PCA
    X_all = StandardScaler().fit_transform(df[CLUSTER_FEATURES].values)

    radar_max   = {s: float(np.percentile(df[s].dropna(), 95)) for s in RADAR_STATS}
    radar_floor = {s: float(np.percentile(df[s].dropna(), 5)) if s in RANGE_STATS else 0.0
                   for s in RADAR_STATS}

    return df, X_all, radar_max, radar_floor


@st.cache_resource
def build_nn(_X):
    nn = NearestNeighbors(metric="euclidean")
    nn.fit(_X)
    return nn


@st.cache_data
def compute_pca(_X):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(_X)
    var = pca.explained_variance_ratio_ * 100
    return coords, var


# ── neighbour web ─────────────────────────────────────────────────────────────

def make_neighbour_web(selected_row, neighbours_df):
    n = min(6, len(neighbours_df))
    BG     = "#0e1117"
    cx, cy = _POSITIONS[0]
    puck_r = 2.046

    fig = go.Figure()

    # Hockey puck watermark
    fig.add_shape(
        type="circle",
        x0=cx - puck_r, y0=cy - puck_r, x1=cx + puck_r, y1=cy + puck_r,
        fillcolor="rgba(20,20,30,0.25)",
        line=dict(color="#555555", width=2.5),
        opacity=0.22,
        xref="x", yref="y", layer="below",
    )
    for r, op in [(puck_r * 0.70, 0.10), (puck_r * 0.40, 0.08)]:
        fig.add_shape(
            type="circle",
            x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color="#666666", width=1, dash="dot"),
            opacity=op,
            xref="x", yref="y", layer="below",
        )

    # Concentric dashed rings
    for r, alpha in [(0.726, 0.30), (1.386, 0.45), (2.046, 0.30)]:
        fig.add_shape(
            type="circle",
            x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color="#7a8aaa", width=1.2, dash="dash"),
            opacity=alpha,
            xref="x", yref="y",
        )

    # Connector lines + distance labels
    for i in range(1, n + 1):
        nx, ny = _POSITIONS[i]
        fig.add_shape(
            type="line",
            x0=cx, y0=cy, x1=nx, y1=ny,
            line=dict(color="#3a3a4a", width=1.2, dash="dash"),
            xref="x", yref="y", layer="below",
        )
        dist = neighbours_df.iloc[i - 1].get("Distance", None)
        if dist is not None:
            frac = 0.50 if ny > cy else 0.65
            mx   = cx + frac * (nx - cx)
            my   = cy + frac * (ny - cy)
            fig.add_annotation(
                x=mx, y=my, text=f"{dist:.2f}",
                showarrow=False,
                font=dict(color="rgba(255,255,255,0.65)", size=10),
                xref="x", yref="y", xanchor="center", yanchor="middle",
                bgcolor="rgba(0,0,0,0)",
            )

    # Centre node
    sel_color = _strat_color(selected_row)
    glow_r = 0.72
    fig.add_shape(
        type="circle",
        x0=cx - glow_r, y0=cy - glow_r, x1=cx + glow_r, y1=cy + glow_r,
        fillcolor=_rgba(sel_color, 0.157),
        line=dict(color=sel_color, width=2),
        xref="x", yref="y",
    )
    fig.add_annotation(
        x=cx, y=cy + 0.12,
        text=f"<b>{_last_name(selected_row['Name'])}</b>",
        showarrow=False,
        font=dict(color="white", size=13),
        xref="x", yref="y", xanchor="center", yanchor="middle",
    )
    fig.add_annotation(
        x=cx, y=cy - 0.26,
        text=f"{selected_row.get('Pos','')} · {selected_row.get('Group','')}",
        showarrow=False,
        font=dict(color="#aaaaaa", size=9),
        xref="x", yref="y", xanchor="center", yanchor="top",
    )
    fig.add_annotation(
        x=cx, y=cy - glow_r - 0.12,
        text=selected_row["Auto_Label"],
        showarrow=False,
        font=dict(color=sel_color, size=10),
        xref="x", yref="y", xanchor="center", yanchor="top",
    )

    # Neighbour nodes
    node_r = 0.52
    for i in range(n):
        row    = neighbours_df.iloc[i]
        px, py = _POSITIONS[i + 1]
        color  = _strat_color(row)
        fig.add_shape(
            type="circle",
            x0=px - node_r, y0=py - node_r, x1=px + node_r, y1=py + node_r,
            fillcolor=_rgba(color, 0.10),
            line=dict(color=color, width=1.5),
            xref="x", yref="y",
        )
        fig.add_annotation(
            x=px, y=py + 0.08,
            text=_last_name(row["Name"]),
            showarrow=False,
            font=dict(color="white", size=10),
            xref="x", yref="y", xanchor="center", yanchor="middle",
        )
        fig.add_annotation(
            x=px, y=py - 0.22,
            text=f"{row.get('Pos','')} · {row.get('Group','')}",
            showarrow=False,
            font=dict(color="#aaaaaa", size=8),
            xref="x", yref="y", xanchor="center", yanchor="top",
        )
        fig.add_annotation(
            x=px, y=py - node_r - 0.12,
            text=row["Auto_Label"],
            showarrow=False,
            font=dict(color=color, size=9),
            xref="x", yref="y", xanchor="center", yanchor="top",
        )

    # Invisible click targets
    click_x       = [_POSITIONS[i][0] for i in range(1, n + 1)]
    click_y       = [_POSITIONS[i][1] for i in range(1, n + 1)]
    hover_text    = []
    border_colors = []
    for i in range(n):
        row   = neighbours_df.iloc[i]
        color = _strat_color(row)
        hover_text.append(
            f"<b><span style='font-size:17px'>{row['Name']}</span></b><br>"
            f"<span style='font-size:12px;color:#aaa'>"
            f"{row.get('Pos','')} · {row.get('Group','')} · {row.get('Nationality','')}</span><br>"
            f"<span style='color:{color}'>{row['Strat_Label']}</span><br>"
            f"<i>Click to compare</i>"
        )
        border_colors.append(color)

    fig.add_trace(go.Scatter(
        x=click_x, y=click_y,
        mode="markers",
        marker=dict(size=72, color="rgba(0,0,0,0)", line=dict(color="rgba(0,0,0,0)", width=0)),
        selected=dict(marker=dict(color="rgba(0,0,0,0)", size=72)),
        unselected=dict(marker=dict(color="rgba(0,0,0,0)", opacity=1)),
        customdata=hover_text,
        hovertemplate="%{customdata}<extra></extra>",
        hoverlabel=dict(
            bgcolor="#1a1a2e",
            bordercolor=border_colors,
            font=dict(color="white", size=14),
            namelength=-1,
        ),
        showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis=dict(range=[-2.50, 2.50], visible=False, fixedrange=True),
        yaxis=dict(range=[-2.60, 2.75], visible=False, fixedrange=True,
                   scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=0, b=0),
        height=648, showlegend=False, hovermode="closest", dragmode=False,
        uirevision=st.session_state.get("_chart_rev", 0),
    )
    return fig


# ── comparison bar ────────────────────────────────────────────────────────────

def comparison_bar_html(label, val_sel, val_cmp, pct_sel, pct_cmp, color_sel, color_cmp, fmt):
    pct_s = pct_sel * 100
    pct_c = pct_cmp * 100
    return (
        f'<div style="margin:5px 0;display:flex;align-items:center;gap:6px">'
        f'  <span style="width:82px;font-size:12px;color:#cccccc">{label}</span>'
        f'  <span style="width:38px;font-size:12px;font-weight:700;text-align:right;color:{color_sel}">'
        f'{val_sel:{fmt}}</span>'
        f'  <span style="flex:1;position:relative;background:#1e1e2a;border-radius:4px;'
        f'height:12px;display:inline-block">'
        f'    <span style="display:block;position:absolute;background:{color_cmp};border-radius:4px;'
        f'           width:{pct_c:.1f}%;height:12px;opacity:0.28"></span>'
        f'    <span style="display:block;position:absolute;background:{color_sel};border-radius:4px;'
        f'           width:{pct_s:.1f}%;height:12px;opacity:0.85"></span>'
        f'    <span style="display:block;position:absolute;left:{pct_c:.1f}%;top:-2px;'
        f'           width:2px;height:16px;background:white;opacity:0.85;'
        f'           transform:translateX(-1px);border-radius:1px"></span>'
        f'  </span>'
        f'  <span style="width:38px;font-size:11px;text-align:left;color:#aaaaaa">'
        f'{val_cmp:{fmt}}</span>'
        f'</div>'
    )


# ── radar chart ───────────────────────────────────────────────────────────────

def make_radar(rows, selected_name, radar_max, radar_floor):
    labels_closed = RADAR_LABELS + [RADAR_LABELS[0]]
    fig = go.Figure()
    for row in rows:
        is_selected = row["Name"] == selected_name
        color = _strat_color(row)
        raw   = [float(row[s]) for s in RADAR_STATS]
        norm  = [_norm(v, s, radar_max, radar_floor) for v, s in zip(raw, RADAR_STATS)]
        hover = (
            f"<b>{row['Name']}</b><br>"
            + "<br>".join(f"{lbl}: {raw[i]:{STAT_FORMATS[s]}}"
                          for i, (lbl, s) in enumerate(zip(RADAR_LABELS, RADAR_STATS)))
            + f"<br>Archetype: {row['Strat_Label']}"
            + "<extra></extra>"
        )
        fig.add_trace(go.Scatterpolar(
            r=norm + [norm[0]], theta=labels_closed,
            fill="toself", fillcolor=color,
            opacity=0.75 if is_selected else 0.18,
            line=dict(color=color, width=3.5 if is_selected else 1.5,
                      dash="solid" if is_selected else "dot"),
            name=row["Name"],
            hovertemplate=hover,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="#f7f7f7",
            radialaxis=dict(visible=True, range=[0, 1],
                            tickvals=[0.25, 0.5, 0.75, 1.0],
                            ticktext=["25%", "50%", "75%", "95th"],
                            tickfont=dict(size=8, color="#aaa"),
                            gridcolor="#ddd", linecolor="#ddd"),
            angularaxis=dict(tickfont=dict(size=12, color="#333"),
                             gridcolor="#ddd", linecolor="#ddd"),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.32,
                    xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(l=70, r=70, t=20, b=110),
        height=480, paper_bgcolor="white",
    )
    return fig


# ── page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NHL Skater Similarity Explorer",
    page_icon="🏒",
    layout="wide",
)

df, X_all, radar_max, radar_floor = load_data()
nn_all = build_nn(X_all)

all_names = sorted(df["Name"].tolist())
f_names   = sorted(df[df["Group"] == "F"]["Name"].tolist())
d_names   = sorted(df[df["Group"] == "D"]["Name"].tolist())

st.title("NHL Skater Similarity Explorer")
st.caption(
    "Select any player to see their 6 nearest neighbours across the combined F+D universe. "
    "Archetypes via position-stratified K-Means (K=4 each) · Combined KNN · PCA projection."
)
st.divider()

st.markdown(
    """<style>
    .js-plotly-plot .plotly .scatterlayer g:last-child .points path {
        cursor: pointer !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🏒 NHL Skater Archetypes")
    st.caption(
        "K-Means clustering on 13 features (2022–23 to 2024–25). "
        "Forwards and defensemen clustered separately. "
        "KNN similarity runs across the combined universe."
    )
    st.divider()

    group          = st.radio("Filter player list", ["Forwards", "Defensemen"])
    n_neighbours   = st.slider("Neighbours to show", 3, 10, 6)
    same_archetype = st.checkbox("Same-archetype only", value=False)

    st.divider()
    st.markdown("**Archetypes**")
    for strat_label, color in ARCHETYPE_COLORS.items():
        st.markdown(
            f'<span style="display:inline-block;width:12px;height:12px;background:{color};'
            f'border-radius:50%;margin-right:6px;vertical-align:middle"></span>'
            f'<span style="font-size:13px">{strat_label}</span>',
            unsafe_allow_html=True,
        )

# ── reset main selector when group filter changes ─────────────────────────────

if st.session_state.get("_prev_group") != group:
    st.session_state.pop("main_select", None)
    st.session_state["_prev_group"] = group

# ── resolve pending click ─────────────────────────────────────────────────────

if "_pending_main" in st.session_state:
    st.session_state["main_select"] = st.session_state.pop("_pending_main")

# ── two-column layout ─────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1])

with col_left:
    sub_main, sub_compare = st.columns(2)

selector_names = f_names if group == "Forwards" else d_names
default_player = "Connor McDavid" if group == "Forwards" else "Cale Makar"
default_idx    = selector_names.index(default_player) if default_player in selector_names else 0

with sub_main:
    selected_name = st.selectbox(
        "Choose a player", selector_names, index=default_idx, key="main_select"
    )

# ── look up selected player + neighbours (combined KNN) ───────────────────────

idx      = df[df["Name"] == selected_name].index[0]
selected = df.loc[idx]
x_sel    = X_all[idx].reshape(1, -1)

distances, indices = nn_all.kneighbors(x_sel, n_neighbors=n_neighbours + 25)
distances, indices = distances[0], indices[0]

mask      = indices != idx
distances = distances[mask]
indices   = indices[mask]

neighbours_df = df.iloc[indices].copy()
neighbours_df["Distance"] = np.round(distances, 3)

if same_archetype:
    neighbours_df = neighbours_df[neighbours_df["Strat_Label"] == selected["Strat_Label"]]

neighbours_df = neighbours_df.head(n_neighbours)
_nb_indices   = neighbours_df.index.tolist()
neighbours_df = neighbours_df.reset_index(drop=True)
neighbours_6  = neighbours_df.head(6).reset_index(drop=True)

# ── dialog ────────────────────────────────────────────────────────────────────

@st.dialog("What would you like to do?")
def neighbour_action_dialog(name):
    row   = df[df["Name"] == name].iloc[0]
    color = _strat_color(row)
    st.markdown(
        f'<div style="background:{_rgba(color, 0.13)};border-left:3px solid {color};'
        f'padding:8px 12px;border-radius:4px;margin-bottom:10px">'
        f'<b style="font-size:16px">{name}</b><br>'
        f'<span style="color:#aaa;font-size:12px">'
        f'{row.get("Pos","")}&nbsp;·&nbsp;{row.get("Group","")}'
        f'&nbsp;·&nbsp;{row.get("Nationality","")}'
        f'</span><br>'
        f'<span style="color:{color};font-size:12px">{row["Strat_Label"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Set as main", use_container_width=True):
            st.session_state["_pending_main"] = name
            st.rerun()
    with c2:
        if st.button("Compare", use_container_width=True):
            st.session_state["compare_select"] = name
            st.rerun()

# ── handle neighbour web click ────────────────────────────────────────────────

_web_state = st.session_state.get("web_chart", {})
_pts       = (_web_state.get("selection") or {}).get("points", [])
_last_pts  = st.session_state.get("_web_last_pts", [])
if _pts and _pts != _last_pts:
    st.session_state["_web_last_pts"] = _pts
    st.session_state["_chart_rev"] = st.session_state.get("_chart_rev", 0) + 1
    _clicked_idx = _pts[0].get("point_index", None)
    if _clicked_idx is not None and _clicked_idx < len(neighbours_6):
        st.session_state["_dialog_player"] = neighbours_6.iloc[_clicked_idx]["Name"]
        st.rerun()

if "_dialog_player" in st.session_state:
    neighbour_action_dialog(st.session_state.pop("_dialog_player"))

# ── neighbour web (right column) ──────────────────────────────────────────────

with col_right:
    fig_web = make_neighbour_web(selected, neighbours_6)
    st.plotly_chart(
        fig_web,
        use_container_width=True,
        on_select="rerun",
        key="web_chart",
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": "reset"},
    )

# ── compare selector + comparison panel (left column) ────────────────────────

with sub_compare:
    compare_name = st.selectbox(
        "Compare with", all_names, index=0, key="compare_select"
    )

sel_color   = _strat_color(selected)
cmp_row     = df[df["Name"] == compare_name].iloc[0]
cmp_color   = _strat_color(cmp_row)
cmp_idx     = df[df["Name"] == compare_name].index[0]
dist_to_sel = float(np.linalg.norm(X_all[cmp_idx] - X_all[idx]))

with col_left:
    def _player_card(row, color, align):
        return (
            f'<div style="flex:1;text-align:{align}">'
            f'  <div style="font-weight:700;font-size:15px;margin-bottom:3px">{row["Name"]}</div>'
            f'  <div style="font-size:12px;color:#aaa;margin-bottom:6px">'
            f'    {row.get("Pos","")}&nbsp;·&nbsp;{row.get("Group","")}'
            f'&nbsp;·&nbsp;{row.get("Nationality","")}'
            f'  </div>'
            f'  <span style="background:{_rgba(color, 0.16)};border:1px solid {color};'
            f'  padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;color:{color}">'
            f'  {row["Auto_Label"]}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0;margin-bottom:14px">'
        f'  {_player_card(selected, sel_color, "left")}'
        f'  <div style="flex:0 0 90px;display:flex;flex-direction:column;'
        f'align-items:center;gap:4px;padding:0 8px">'
        f'    <div style="width:1px;height:20px;background:#444"></div>'
        f'    <div style="font-size:11px;color:#aaa;white-space:nowrap;text-align:center">'
        f'      dist<br><b style="color:white;font-size:13px">{dist_to_sel:.3f}</b>'
        f'    </div>'
        f'    <div style="width:1px;height:20px;background:#444"></div>'
        f'  </div>'
        f'  {_player_card(cmp_row, cmp_color, "right")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="font-size:12px;color:#cccccc;margin-bottom:6px">'
        f'<span style="color:{sel_color}">&#9632;</span> {selected_name}&nbsp;&nbsp;'
        f'<span style="color:{cmp_color};opacity:0.5">&#9632;</span> {compare_name}'
        f'</div>',
        unsafe_allow_html=True,
    )

    for lbl, stat in zip(RADAR_LABELS, RADAR_STATS):
        v_sel = float(selected[stat]) if pd.notna(selected[stat]) else 0.0
        v_cmp = float(cmp_row[stat])  if pd.notna(cmp_row[stat])  else 0.0
        p_sel = _norm(v_sel, stat, radar_max, radar_floor)
        p_cmp = _norm(v_cmp, stat, radar_max, radar_floor)
        st.markdown(
            comparison_bar_html(lbl, v_sel, v_cmp, p_sel, p_cmp,
                                sel_color, cmp_color, STAT_FORMATS[stat]),
            unsafe_allow_html=True,
        )

# ── PCA scatter (combined universe, position-stratified colours) ──────────────

pca_coords, pca_var = compute_pca(X_all)

fig_pca = go.Figure()
for strat_label, color in ARCHETYPE_COLORS.items():
    mask   = df["Strat_Label"] == strat_label
    if not mask.any():
        continue
    sub        = df[mask]
    sub_coords = pca_coords[mask]
    grp        = strat_label[0]
    symbol     = "circle" if grp == "F" else "square"
    hover = [
        f"<b>{row['Name']}</b><br>"
        f"{row.get('Pos','')} · {row.get('Group','')} · {row.get('Nationality','')}<br>"
        f"Archetype: {row['Strat_Label']}<br>"
        + "<br>".join(f"{lbl}: {row[s]:{STAT_FORMATS[s]}}"
                      for lbl, s in zip(RADAR_LABELS, RADAR_STATS))
        for _, row in sub.iterrows()
    ]
    fig_pca.add_trace(go.Scatter(
        x=sub_coords[:, 0], y=sub_coords[:, 1],
        mode="markers", name=strat_label,
        marker=dict(color=color, size=7, opacity=0.75,
                    symbol=symbol, line=dict(width=0)),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
    ))

nb_coords = pca_coords[_nb_indices]
nb_hover  = [
    f"<b>{row['Name']}</b><br>{row['Strat_Label']}<br>Distance: {row['Distance']:.3f}"
    for _, row in neighbours_df.iterrows()
]
fig_pca.add_trace(go.Scatter(
    x=nb_coords[:, 0], y=nb_coords[:, 1],
    mode="markers", name="Neighbours",
    marker=dict(color="white", size=11, opacity=0.9,
                symbol="circle-open", line=dict(color="white", width=2)),
    hovertemplate="%{customdata}<extra></extra>",
    customdata=nb_hover,
))
fig_pca.add_trace(go.Scatter(
    x=[pca_coords[idx, 0]], y=[pca_coords[idx, 1]],
    mode="markers+text", name=selected_name,
    marker=dict(color=sel_color, size=14, symbol="star",
                line=dict(color="white", width=1.5)),
    text=[selected_name], textposition="top center",
    textfont=dict(color="white", size=12),
    hovertemplate=f"<b>{selected_name}</b><br>{selected['Strat_Label']}<extra></extra>",
))
fig_pca.update_layout(
    xaxis_title=f"PC1 ({pca_var[0]:.1f}% variance)",
    yaxis_title=f"PC2 ({pca_var[1]:.1f}% variance)",
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font=dict(color="white"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(gridcolor="#2a2a3a", zerolinecolor="#2a2a3a"),
    yaxis=dict(gridcolor="#2a2a3a", zerolinecolor="#2a2a3a"),
    height=580, margin=dict(l=20, r=20, t=20, b=20),
    hovermode="closest",
)
st.plotly_chart(fig_pca, use_container_width=True)

# ── neighbour table ───────────────────────────────────────────────────────────

st.divider()
st.markdown(f"#### {n_neighbours} Nearest Neighbours")

display_cols = ["Name", "Group", "Pos", "Auto_Label", "GP",
                "G_60", "A_60", "SOG_60", "xGF_pct_5v5",
                "HIT_60", "BLK_60", "Fights", "Distance"]
display_cols = [c for c in display_cols if c in neighbours_df.columns]
display_df   = neighbours_df[display_cols].rename(columns={"Auto_Label": "Archetype"})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Distance":    st.column_config.NumberColumn(format="%.3f"),
        "GP":          st.column_config.NumberColumn(format="%d"),
        "Fights":      st.column_config.NumberColumn(format="%d"),
        "G_60":        st.column_config.ProgressColumn("G/60",       min_value=0,  max_value=2.0,  format="%.2f"),
        "A_60":        st.column_config.ProgressColumn("A/60",       min_value=0,  max_value=2.5,  format="%.2f"),
        "SOG_60":      st.column_config.ProgressColumn("SOG/60",     min_value=0,  max_value=12.0, format="%.2f"),
        "xGF_pct_5v5": st.column_config.ProgressColumn("xGF% (5v5)", min_value=40, max_value=65,   format="%.1f"),
        "HIT_60":      st.column_config.ProgressColumn("Hits/60",    min_value=0,  max_value=15.0, format="%.2f"),
        "BLK_60":      st.column_config.ProgressColumn("Blocks/60",  min_value=0,  max_value=8.0,  format="%.2f"),
    },
)
