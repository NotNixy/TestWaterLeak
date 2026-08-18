"""
PAIP Non-Revenue Water — Leakage Intervention Priority Dashboard
================================================================
Turns PAIP's published monthly production and billing figures into a ranked,
volume-weighted repair schedule.

Run:  streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr, kendalltau

import theme as theme_mod
import train_models as tm

DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="PAIP NRW — Intervention Priority",
                   page_icon="◔", layout="wide",
                   initial_sidebar_state="expanded")


def detected_mode() -> str:
    """Read the browser/OS colour scheme. Streamlit exposes the active theme
    through st.context; when no theme is pinned in config.toml that follows
    `prefers-color-scheme`, which is what makes Auto track the system."""
    try:
        t = getattr(st.context, "theme", None)
        val = getattr(t, "type", None) if t is not None else None
        if val in ("light", "dark"):
            return val
    except Exception:
        pass
    return "light"


_detected = detected_mode()
with st.sidebar:
    _pref = st.radio("Appearance", ["Auto", "Light", "Dark"], index=0,
                     horizontal=True,
                     help=f"Auto follows your system setting "
                          f"(currently detected: {_detected}).")
MODE = theme_mod.resolve_mode(_pref, _detected)
T = theme_mod.Theme(MODE)
st.markdown(T.css, unsafe_allow_html=True)

PLOT_CFG = {"displayModeBar": False, "responsive": True}

LIPS_COMPONENTS = {
    "physical_loss_m3":   ("Physical loss volume",  "m³ of leakage a repair physically recovers"),
    "nrw_per_km_m3":      ("Leak concentration",    "NRW per km of main — water recovered per crew-day"),
    "nrw_pct":            ("Loss rate",             "NRW as % of production — the efficiency signal"),
    "bursts_per_100km":   ("Burst frequency",       "Recorded bursts per 100 km — observed deterioration"),
    "asset_age_index":    ("Asset condition",       "Composite of plant age (60%) and meter age (40%)"),
    "physical_share_pct": ("Repairable share",      "Share of loss that is physical rather than commercial"),
    "criticality":        ("Criticality (model)",   "Model-based index: unexplained loss, deterioration and trend"),
}
DEFAULT_WEIGHTS = {"physical_loss_m3": 35, "nrw_per_km_m3": 15, "nrw_pct": 13,
                   "bursts_per_100km": 12, "asset_age_index": 15,
                   "physical_share_pct": 10}


# ==========================================================================
# Data
# ==========================================================================

@st.cache_data
def load():
    m = pd.read_csv(DATA / "nrw_plant_month.csv", parse_dates=["date"])
    y = pd.read_csv(DATA / "nrw_plant_year.csv")
    x = pd.read_csv(DATA / "plant_crosswalk.csv")
    q = pd.read_csv(DATA / "data_quality.csv")
    v = pd.read_csv(DATA / "missing_values.csv")
    return m, y, x, q, v


@st.cache_data
def load_ml():
    """Model artefacts produced by train_models.py. Training is done offline so
    the outputs are reproducible and auditable rather than refitted per click."""
    try:
        p = pd.read_csv(DATA / "ml_plant.csv")
        mm = pd.read_csv(DATA / "ml_monthly.csv", parse_dates=["date"])
        met = json.loads((DATA / "model_metrics.json").read_text())
        return p, mm, met
    except FileNotFoundError:
        return None, None, None


@st.cache_data
def cluster_at(k: int, yr: int):
    """Re-run KMeans at a chosen k for a given year. Cheap (74 rows), and
    separation is modest at every k, so the operator should be able to try
    alternatives."""
    py = yearly[yearly.year == yr].copy()
    scored, profile, sil, best = tm.archetypes(py, k=k)
    return scored[["plant", "cluster", "archetype"]], profile, sil, best


@st.cache_data
def load_coverage():
    p = DATA / "year_coverage.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


monthly, yearly, crosswalk, quality, missing = load()
ml_plant, ml_monthly, ml_metrics = load_ml()
HAS_ML = ml_plant is not None
coverage = load_coverage()

# Everything below derives its year range from the data, so a refresh that adds
# 2026 needs no code change.
YEARS = sorted(int(y) for y in monthly.year.unique())
YEAR_MIN, YEAR_MAX = YEARS[0], YEARS[-1]
YEAR_SPAN = f"{YEAR_MIN}" if YEAR_MIN == YEAR_MAX else f"{YEAR_MIN}–{YEAR_MAX}"
ML_YEAR = int(ml_metrics.get("focus_year", YEAR_MAX)) if HAS_ML else None

# Months observed per year, so partial years can be labelled and annualised.
if coverage is not None:
    MONTHS_BY_YEAR = dict(zip(coverage.year.astype(int), coverage.months.astype(int)))
else:
    MONTHS_BY_YEAR = (monthly.groupby("year").date.apply(lambda s: s.dt.month.nunique())
                      .astype(int).to_dict())


def year_label(y: int) -> str:
    m = MONTHS_BY_YEAR.get(int(y), 12)
    return f"{int(y)}" if m >= 12 else f"{int(y)} · {m} of 12 months"


def is_partial(y: int) -> bool:
    return MONTHS_BY_YEAR.get(int(y), 12) < 12


def percentile_rank(s):
    return s.rank(pct=True, method="average") * 100


@st.cache_data
def score_lips(df: pd.DataFrame, weights: tuple) -> pd.DataFrame:
    """Recompute LIPS for the current weight vector. Percentile ranks are taken
    within the filtered cohort so the score always reflects the plants on screen."""
    w = dict(weights)
    total = sum(w.values()) or 1
    out = df.copy()
    score = pd.Series(0.0, index=out.index)
    for col, wt in w.items():
        pr = percentile_rank(out[col])
        out[f"pr_{col}"] = pr
        score += pr * (wt / total)
    out["lips"] = score.round(2)
    # A repair queue must be a strict order: two plants cannot both be
    # "priority 31". Ties on LIPS are broken by physical loss volume, so the
    # plant with more recoverable water is visited first.
    out = out.sort_values(["lips", "physical_loss_m3"], ascending=False)
    out["lips_rank"] = np.arange(1, len(out) + 1)
    out["volume_rank"] = out.nrw_m3.rank(ascending=False, method="first").astype(int)
    out["rate_rank"] = out.nrw_pct.rank(ascending=False, method="first").astype(int)
    out["rank_gap"] = out.rate_rank - out.volume_rank
    return out


def fmt(n, dp=0):
    return f"{n:,.{dp}f}"


def m3(n):
    """Volumes span five orders of magnitude, so units are scaled per value.
    Uppercase M for millions — a lowercase 'm' beside 'm³' reads as metres."""
    if abs(n) >= 1e9:
        return f"{n/1e9:,.2f}B"
    if abs(n) >= 1e6:
        return f"{n/1e6:,.1f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:,.0f}k"
    return f"{n:,.0f}"


def rm(n):
    if abs(n) >= 1e9:
        return f"{n/1e9:,.2f}B"
    if abs(n) >= 1e6:
        return f"{n/1e6:,.1f}M"
    return f"{n:,.0f}"


# ==========================================================================
# Sidebar
# ==========================================================================


with st.sidebar:
    st.markdown("### Pengurusan Air Pahang Berhad")
    st.markdown(f'<div class="caption">Non-Revenue Water intervention '
                f'targeting · {monthly.plant.nunique()} plants · '
                f'{monthly.district.nunique()} districts · {YEAR_SPAN}</div>',
                unsafe_allow_html=True)

    year = st.selectbox("Year", sorted(YEARS, reverse=True), index=0,
                        format_func=year_label)
    if is_partial(year):
        st.markdown(
            f'<div class="caption">⚠ {year} is still in progress '
            f'({MONTHS_BY_YEAR[year]} of 12 months). Volume totals are actuals; '
            f'charts that compare years annualise them. Rates are unaffected.'
            f'</div>', unsafe_allow_html=True)

    regions = st.multiselect("Region", sorted(yearly.region.unique()),
                             default=sorted(yearly.region.unique()))
    districts_all = sorted(yearly[yearly.region.isin(regions)].district.unique())
    districts = st.multiselect("District", districts_all, default=districts_all)
    areas = st.multiselect("Area type", sorted(yearly.area_type.unique()),
                           default=sorted(yearly.area_type.unique()))

    st.markdown("---")
    st.markdown("#### LIPS weights")
    st.markdown('<div class="caption">Relative importance of each component. '
                'Values are renormalised to sum to 100, so only the ratios '
                'matter. Set a weight to 0 to drop a component.</div>',
                unsafe_allow_html=True)

    weights = {}
    for col, default in DEFAULT_WEIGHTS.items():
        label, help_text = LIPS_COMPONENTS[col]
        weights[col] = st.slider(label, 0, 50, default, 1, help=help_text)

    blend_crit = False
    crit_weight = 0
    if HAS_ML:
        st.markdown("---")
        st.markdown("#### Model input")
        blend_crit = st.checkbox(
            "Fold Criticality into LIPS", value=False,
            help="Adds the model-based Criticality Index as a seventh LIPS "
                 "component. Off by default so the deterministic score and the "
                 "model-based one can be compared side by side.")
        crit_weight = st.slider("Criticality weight", 0, 50, 20, 1,
                                disabled=not blend_crit)

    if st.button("Reset to defaults", width='stretch'):
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="caption">Source: PAIP monthly production, billing '
                f'and operations records, {YEAR_SPAN}.</div>',
                unsafe_allow_html=True)


mask = (yearly.year == year) & yearly.region.isin(regions) \
       & yearly.district.isin(districts) & yearly.area_type.isin(areas)
sel = yearly[mask].copy()

mmask = (monthly.year == year) & monthly.region.isin(regions) \
        & monthly.district.isin(districts) & monthly.area_type.isin(areas)
msel = monthly[mmask].copy()

if sel.empty:
    st.error("No plants match the current filters. Widen the selection in the sidebar.")
    st.stop()

# Attach model outputs before scoring, so criticality can act as a LIPS
# component when the operator asks for it.
#
# The models are fitted for one focus year (the latest in the data). Merging
# those scores onto a different year would silently mislabel them, so they are
# attached only when the selected year matches; otherwise the columns are
# present but empty and the Early Warning tab explains why.
ML_COLS = ["criticality", "criticality_rank", "unexplained_pp",
           "unexplained_m3", "expected_nrw_pct", "actual_nrw_pct",
           "trend_pp_yr", "trend_p", "trend_recent_pp_yr", "step_shift_pp",
           "step_p", "anomaly_months", "worst_z", "anomaly_score",
           "archetype", "cluster", "projected_nrw_pct_12m",
           "projected_extra_m3", "volatility_pp", "latest_nrw_pct",
           "pr_unexplained", "pr_deterioration", "pr_trend"]
ML_MATCHES_YEAR = HAS_ML and year == ML_YEAR
if ML_MATCHES_YEAR:
    sel = sel.merge(ml_plant[["plant"] + ML_COLS], on="plant", how="left")
elif HAS_ML:
    for c in ML_COLS:
        sel[c] = np.nan

lips_weights = dict(weights)
# Only blend when the model actually covers the selected year.
if ML_MATCHES_YEAR and blend_crit and crit_weight > 0:
    lips_weights["criticality"] = crit_weight

sel_plain = score_lips(sel, tuple(sorted(weights.items())))
sel = score_lips(sel, tuple(sorted(lips_weights.items())))

# System-level aggregates for the current selection.
tot_prod = sel.production_m3.sum()
tot_nrw = sel.nrw_m3.sum()
tot_val = sel.nrw_value_rm.sum()
tot_phys = sel.physical_loss_m3.sum()
sys_pct = tot_nrw / tot_prod * 100
n_plants = len(sel)

prev = yearly[(yearly.year == year - 1) & yearly.plant.isin(sel.plant)]
prev_pct = (prev.nrw_m3.sum() / prev.production_m3.sum() * 100) if len(prev) else np.nan


# ==========================================================================
# Header
# ==========================================================================

st.markdown("## Non-Revenue Water — Intervention Priority")
st.markdown(
    f'<div class="caption">Every cubic metre lost has already absorbed the full '
    f'cost of abstraction, treatment, chemicals and pumping. This dashboard '
    f'ranks where repair crews recover the most water — not where the loss '
    f'percentage looks worst. <b>{n_plants} plants · {year}</b></div>',
    unsafe_allow_html=True)

k = st.columns(4)
delta = ""
if not np.isnan(prev_pct):
    d = sys_pct - prev_pct
    cls = "tile-delta-good" if d < 0 else "tile-delta-bad"
    arrow = "↓" if d < 0 else "↑"
    delta = f'<span class="{cls}">{arrow} {abs(d):.2f} pp</span> vs {year-1}'

k[0].markdown(T.tile("System loss rate", f"{sys_pct:.1f}", "%",
                     delta or f"Production-weighted across {n_plants} plants"),
              unsafe_allow_html=True)
k[1].markdown(T.tile("Water lost", m3(tot_nrw), "m³",
                     f"{m3(tot_nrw/365)} m³ every day"), unsafe_allow_html=True)

k[2].markdown(T.tile("Physical leakage", f"{tot_phys/tot_nrw*100:.0f}", "%",
                     f"{m3(tot_phys)} m³ addressable by pipe repair"),
              unsafe_allow_html=True)
above = int((sel.nrw_pct > T.POLICY_TARGET_PCT).sum())
k[3].markdown(T.tile("Above 25% target", f"{above}", f"of {n_plants}",
                     f"{above/n_plants*100:.0f}% of plants exceed national policy target"),
              unsafe_allow_html=True)

st.markdown("")

tabs = st.tabs(["Overview", "Rate vs Volume", "Priority Schedule",
                "Loss Composition", "Plant Profile"])
(TAB_OVERVIEW, TAB_RATEVOL, TAB_SCHEDULE,
 TAB_COMPOSITION, TAB_PLANT) = tabs


# ==========================================================================
# TAB 1 — Overview
# ==========================================================================
with TAB_OVERVIEW:
    c1, c2 = st.columns([1.15, 1])

    with c1:
        st.markdown("#### Where the water goes")
        dist = (sel.groupby("district", as_index=False)
                   .agg(nrw_m3=("nrw_m3", "sum"),
                        production_m3=("production_m3", "sum"),
                        plants=("plant", "count"),
                        value_rm=("nrw_value_rm", "sum"))
                   .assign(nrw_pct=lambda d: d.nrw_m3 / d.production_m3 * 100)
                   .sort_values("nrw_m3", ascending=True))

        fig = go.Figure(go.Bar(
            x=dist.nrw_m3, y=dist.district, orientation="h",
            marker=dict(color=dist.nrw_pct, colorscale=T.SEQ,
                        line=dict(color=T.SURFACE, width=2),
                        colorbar=dict(title=dict(text="Loss rate %", side="right",
                                                 font=dict(size=11, color=T.INK_2)),
                                      thickness=10, len=0.55, x=1.01,
                                      outlinewidth=0, ticks="outside",
                                      ticklen=3, tickcolor=T.BASELINE,
                                      tickfont=dict(size=10, color=T.MUTED))),
            text=[m3(v) for v in dist.nrw_m3], textposition="outside",
            textfont=dict(size=11, color=T.INK_2),
            customdata=np.stack([dist.nrw_pct, dist.plants, dist.value_rm], -1),
            hovertemplate=("<b>%{y}</b><br>NRW volume  %{x:,.0f} m³<br>"
                           "Loss rate  %{customdata[0]:.1f}%<br>"
                           "Plants  %{customdata[1]}<br>"
                           "Value  RM %{customdata[2]:,.0f}<extra></extra>")))
        fig.update_layout(
            title="NRW volume by district, shaded by loss rate",
            height=430, bargap=0.35,
            xaxis=dict(title="NRW volume (m³)", range=[0, dist.nrw_m3.max() * 1.16]),
            yaxis=dict(title=None))
        st.plotly_chart(fig, width='stretch', config=PLOT_CFG, theme=None)
        st.markdown(
            '<div class="caption">Colour carries the loss rate, length carries '
            'the volume. Districts that look dark but short are efficient in '
            'neither sense that matters operationally — they simply have little '
            'water to recover.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("#### Loss concentration")
        sv = sel.sort_values("nrw_m3", ascending=False).reset_index(drop=True)
        sv["cum_share"] = sv.nrw_m3.cumsum() / sv.nrw_m3.sum() * 100
        sv["plant_n"] = np.arange(1, len(sv) + 1)
        n10 = min(10, len(sv))
        share10 = sv.loc[n10 - 1, "cum_share"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sv.plant_n, y=sv.cum_share, mode="lines",
            line=dict(color=T.BLUE, width=2.5), fill="tozeroy",
            fillcolor=T.TILE_WASH, name="Cumulative share",
            hovertemplate=("Top %{x} plants<br>hold %{y:.1f}% of all NRW"
                           "<extra></extra>")))
        fig.add_hline(y=share10, line=dict(color=T.BASELINE, width=1, dash="dot"))
        fig.add_vline(x=n10, line=dict(color=T.BASELINE, width=1, dash="dot"))
        fig.add_annotation(x=n10, y=share10, text=f"<b>Top {n10} plants → {share10:.0f}%</b>",
                           showarrow=True, arrowhead=0, arrowwidth=1,
                           arrowcolor=T.MUTED, ax=58, ay=34,
                           font=dict(size=12, color=T.INK), bgcolor=T.SURFACE,
                           bordercolor=T.BASELINE, borderwidth=1, borderpad=5)
        fig.update_layout(
            title="Cumulative share of total NRW, plants ranked by volume",
            height=430, showlegend=False,
            xaxis=dict(title="Plants, largest loss first"),
            yaxis=dict(title="Cumulative % of NRW", range=[0, 102],
                       ticksuffix="%"))
        st.plotly_chart(fig, width='stretch', config=PLOT_CFG, theme=None)
        st.markdown(
            f'<div class="caption">Losses are heavily concentrated: {share10:.0f}% '
            f'of all non-revenue water sits in {n10} of {n_plants} plants. A crew '
            f'programme that never leaves this group addresses most of the '
            f'recoverable volume.</div>', unsafe_allow_html=True)

    st.markdown("---")
    # The Seasonality panel that used to fill the left column was removed, so
    # the trajectory chart now spans the full width instead of sitting in a
    # half column beside empty space.
    c4 = st.container()
    with c4:
        st.markdown(f"#### {len(YEARS)}-year trajectory")
        tr = (yearly[yearly.plant.isin(sel.plant)]
              .groupby("year", as_index=False)
              .agg(nrw_m3=("nrw_m3", "sum"), production_m3=("production_m3", "sum"))
              .assign(nrw_pct=lambda d: d.nrw_m3 / d.production_m3 * 100))
        # The rate is a ratio, so a partial year needs no annualisation here —
        # but it does need labelling, or a half-finished year reads as a result.
        tr["partial"] = tr.year.map(is_partial)
        tr["tick"] = tr.year.astype(int).astype(str) + np.where(
            tr.partial, "*", "")

        fig = go.Figure()
        fig.add_hrect(y0=0, y1=T.POLICY_TARGET_PCT, fillcolor="rgba(12,163,12,0.06)",
                      line_width=0)
        fig.add_hline(y=T.POLICY_TARGET_PCT,
                      line=dict(color=T.GOOD, width=1.5, dash="dash"),
                      annotation_text="Policy target 25%",
                      annotation_position="bottom right",
                      annotation_font=dict(size=11, color=T.SUCCESS_TEXT))
        fig.add_hline(y=T.NATIONAL_NRW_PCT,
                      line=dict(color=T.MUTED, width=1.5, dash="dot"),
                      annotation_text="National average 35%",
                      annotation_position="top right",
                      annotation_font=dict(size=11, color=T.MUTED))
        fig.add_trace(go.Bar(
            x=tr.tick, y=tr.nrw_pct, name="PAIP selection",
            # A partial year is drawn in the "serious" status colour and starred,
            # so it can never be mistaken for a settled annual result.
            marker=dict(color=[T.SERIOUS if p else T.BLUE for p in tr.partial],
                        line=dict(color=T.SURFACE, width=2)),
            width=0.45, text=[f"{v:.1f}%" for v in tr.nrw_pct],
            textposition="outside", textfont=dict(size=12, color=T.INK),
            customdata=np.stack([tr.nrw_m3, tr.year.map(
                lambda y: MONTHS_BY_YEAR.get(int(y), 12))], -1),
            hovertemplate=("<b>%{x}</b><br>Loss rate  %{y:.2f}%<br>"
                           "Volume  %{customdata[0]:,.0f} m³<br>"
                           "Months observed  %{customdata[1]} of 12"
                           "<extra></extra>")))
        fig.update_layout(
            title="Loss rate against national reference points", height=340,
            showlegend=False, xaxis=dict(title=None, tickmode="array",
                                         tickvals=tr.tick.tolist()),
            yaxis=dict(title="NRW (% of production)", ticksuffix="%",
                       range=[0, max(45, tr.nrw_pct.max() * 1.25)]))
        st.plotly_chart(fig, width='stretch', config=PLOT_CFG, theme=None)
        if tr.partial.any():
            st.markdown(
                '<div class="caption">* Year still in progress. The rate is a '
                'ratio so it stays comparable, but it reflects only the months '
                'observed so far.</div>', unsafe_allow_html=True)

        if len(tr) >= 2:
            rate_change = (tr.nrw_pct.iloc[-1] - tr.nrw_pct.iloc[0]) / (len(tr) - 1)
            gap = tr.nrw_pct.iloc[-1] - T.POLICY_TARGET_PCT
            if rate_change < -0.01:
                yrs = gap / abs(rate_change)
                msg = (f'Improving at <b>{abs(rate_change):.2f} pp per year</b>. '
                       f'Held at that pace, the {gap:.1f} pp gap to the 25% '
                       f'policy target closes in roughly <b>{yrs:.0f} years</b>. '
                       f'Targeted intervention is what shortens that.')
            else:
                msg = (f'The rate is not improving. The gap to the 25% policy '
                       f'target stands at <b>{gap:.1f} pp</b>.')
            st.markdown(f'<div class="caption">{msg}</div>', unsafe_allow_html=True)

    with st.expander("District table"):
        show = dist.sort_values("nrw_m3", ascending=False)[
            ["district", "plants", "production_m3", "nrw_m3", "nrw_pct", "value_rm"]]
        st.dataframe(show, width='stretch', hide_index=True,
                     column_config={
                         "district": "District", "plants": "Plants",
                         "production_m3": st.column_config.NumberColumn("Production m³", format="%,d"),
                         "nrw_m3": st.column_config.NumberColumn("NRW m³", format="%,d"),
                         "nrw_pct": st.column_config.NumberColumn("Loss rate", format="%.1f%%"),
                         "value_rm": st.column_config.NumberColumn("Value RM", format="%,d")})


# ==========================================================================
# TAB 2 — Rate vs Volume
# ==========================================================================
with TAB_RATEVOL:
    st.markdown("#### Two measures, two different repair queues")

    rho = spearmanr(sel.nrw_pct, sel.nrw_m3).statistic
    tau = kendalltau(sel.rate_rank, sel.volume_rank).statistic
    n_top = min(10, n_plants)
    top_rate = sel.nsmallest(n_top, "rate_rank")
    top_vol = sel.nsmallest(n_top, "volume_rank")
    overlap = len(set(top_rate.plant) & set(top_vol.plant))
    w_rate, w_vol = top_rate.nrw_m3.sum(), top_vol.nrw_m3.sum()
    ratio = w_vol / w_rate if w_rate else np.nan

    st.markdown(T.callout(
        f"Ranking the same {n_plants} plants by loss <b>rate</b> and by loss "
        f"<b>volume</b> produces a rank correlation of <b>ρ = {rho:.2f}</b> "
        f"(Kendall τ = {tau:.2f}). The two top-{n_top} queues share "
        f"<b>{overlap} plant{'s' if overlap != 1 else ''}</b>. The volume queue "
        f"covers <b>{m3(w_vol)} m³</b> of losses against <b>{m3(w_rate)} m³</b> "
        f"for the rate queue — <b>{ratio:.1f}× more water</b> for the same ten "
        f"crew deployments.",
        "crit" if overlap <= 2 else "warn"), unsafe_allow_html=True)

    c1, c2 = st.columns([1.25, 1])

    with c1:
        pl = sel.copy()
        cond = [pl.plant.isin(set(top_rate.plant) & set(top_vol.plant)),
                pl.plant.isin(top_vol.plant), pl.plant.isin(top_rate.plant)]
        pl["queue"] = np.select(
            cond, [f"Both queues", f"Top {n_top} by volume", f"Top {n_top} by rate"],
            default="Neither")

        order = [f"Top {n_top} by volume", f"Top {n_top} by rate", "Both queues", "Neither"]
        colors = {f"Top {n_top} by volume": T.BLUE, f"Top {n_top} by rate": T.ORANGE,
                  "Both queues": T.AQUA, "Neither": T.NEUTRAL}

        fig = go.Figure()
        for grp in order:
            g = pl[pl.queue == grp]
            if g.empty:
                continue
            fig.add_trace(go.Scatter(
                x=g.production_m3, y=g.nrw_pct, mode="markers", name=grp,
                marker=dict(size=np.sqrt(g.nrw_m3 / pl.nrw_m3.max()) * 44 + 7,
                            color=colors[grp], opacity=0.85,
                            line=dict(color=T.SURFACE, width=2)),
                customdata=np.stack([g.plant, g.district, g.nrw_m3,
                                     g.volume_rank, g.rate_rank], -1),
                hovertemplate=("<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                               "Production  %{x:,.0f} m³<br>"
                               "Loss rate  %{y:.1f}%  (rank %{customdata[4]})<br>"
                               "NRW volume  %{customdata[2]:,.0f} m³  "
                               "(rank %{customdata[3]})<extra></extra>")))

        # Direct-label only the largest few, offset below each bubble by its own
        # radius so the text clears the mark.
        lab = pl.nlargest(3, "nrw_m3")
        for _, r in lab.iterrows():
            radius = (np.sqrt(r.nrw_m3 / pl.nrw_m3.max()) * 44 + 7) / 2
            fig.add_annotation(x=np.log10(r.production_m3), y=r.nrw_pct,
                               text=r.plant, showarrow=False,
                               yshift=-(radius + 14),
                               font=dict(size=10.5, color=T.INK_2),
                               bgcolor=T.SURFACE, opacity=0.9, borderpad=2)
        fig.add_hline(y=T.POLICY_TARGET_PCT,
                      line=dict(color=T.GOOD, width=1.2, dash="dash"),
                      annotation_text="25% target",
                      annotation_position="top left",
                      annotation_font=dict(size=10.5, color=T.SUCCESS_TEXT))
        fig.update_layout(
            title="Loss rate against plant size — bubble area is NRW volume",
            height=520,
            xaxis=dict(title="Annual production (m³, log scale)", type="log",
                       dtick=1, minor=dict(showgrid=False)),
            yaxis=dict(title="Loss rate (% of production)", ticksuffix="%"))
        st.plotly_chart(fig, width='stretch', config=PLOT_CFG, theme=None)
        st.markdown(
            '<div class="caption">The orange plants post the alarming '
            'percentages; the blue plants hold the water. They are almost '
            'entirely different sets, because a percentage is a ratio to plant '
            'size — small plants reach extreme rates on modest volumes.</div>',
            unsafe_allow_html=True)

    with c2:
        st.markdown("###### How far plants move between the two rankings")
        # A dumbbell rather than a slope chart: giving every plant its own row
        # keeps the labels legible, where a two-column slope chart packs volume
        # ranks 1..n on top of each other.
        n_dumb = min(12, n_plants)
        mv = sel.nsmallest(n_dumb, "volume_rank")[
            ["plant", "rate_rank", "volume_rank", "nrw_m3", "nrw_pct"]].copy()
        mv = mv.sort_values("volume_rank", ascending=False)

        fig = go.Figure()
        for _, r in mv.iterrows():
            fig.add_trace(go.Scatter(
                x=[r.volume_rank, r.rate_rank], y=[r.plant, r.plant],
                mode="lines", line=dict(color=T.NEUTRAL, width=2.5),
                showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=mv.volume_rank, y=mv.plant, mode="markers", name="Rank by volume",
            marker=dict(size=11, color=T.BLUE,
                        line=dict(color=T.SURFACE, width=2)),
            customdata=mv.nrw_m3,
            hovertemplate=("<b>%{y}</b><br>Volume rank  %{x}<br>"
                           "NRW  %{customdata:,.0f} m³<extra></extra>")))
        fig.add_trace(go.Scatter(
            x=mv.rate_rank, y=mv.plant, mode="markers+text", name="Rank by rate",
            marker=dict(size=11, color=T.ORANGE,
                        line=dict(color=T.SURFACE, width=2)),
            text=[f"  {v}" for v in mv.rate_rank], textposition="middle right",
            textfont=dict(size=10.5, color=T.MUTED),
            customdata=mv.nrw_pct,
            hovertemplate=("<b>%{y}</b><br>Rate rank  %{x}<br>"
                           "Loss rate  %{customdata:.1f}%<extra></extra>")))
        fig.update_layout(
            title=f"The {len(mv)} largest-volume plants in each ranking",
            height=520,
            xaxis=dict(title="Rank among all plants (1 = highest priority)",
                       range=[0, n_plants + 5]),
            yaxis=dict(title=None, tickfont=dict(size=11)))
        st.plotly_chart(fig, width='stretch', config=PLOT_CFG, theme=None)
        st.markdown(
            '<div class="caption">Each row is one plant; the bar spans the two '
            'rankings. Long bars are plants the rate ranking buries — large, '
            'apparently acceptable performers that quietly lose the most '
            'water.</div>', unsafe_allow_html=True)

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"###### Queue A — top {n_top} by loss rate")
        st.dataframe(
            top_rate[["rate_rank", "plant", "district", "nrw_pct", "nrw_m3", "volume_rank"]],
            width='stretch', hide_index=True, height=390,
            column_config={
                "rate_rank": st.column_config.NumberColumn("#", width="small"),
                "plant": "Plant", "district": "District",
                "nrw_pct": st.column_config.NumberColumn("Loss rate", format="%.1f%%"),
                "nrw_m3": st.column_config.NumberColumn("NRW m³", format="%,d"),
                "volume_rank": st.column_config.NumberColumn("Volume rank")})
        st.markdown(f'<div class="caption">Total water in this queue: '
                    f'<b>{fmt(w_rate)} m³</b></div>', unsafe_allow_html=True)

    with c4:
        st.markdown(f"###### Queue B — top {n_top} by loss volume")
        st.dataframe(
            top_vol[["volume_rank", "plant", "district", "nrw_m3", "nrw_pct", "rate_rank"]],
            width='stretch', hide_index=True, height=390,
            column_config={
                "volume_rank": st.column_config.NumberColumn("#", width="small"),
                "plant": "Plant", "district": "District",
                "nrw_m3": st.column_config.NumberColumn("NRW m³", format="%,d"),
                "nrw_pct": st.column_config.NumberColumn("Loss rate", format="%.1f%%"),
                "rate_rank": st.column_config.NumberColumn("Rate rank")})
        st.markdown(f'<div class="caption">Total water in this queue: '
                    f'<b>{fmt(w_vol)} m³</b> — {ratio:.1f}× Queue A</div>',
                    unsafe_allow_html=True)


# ==========================================================================
# TAB 3 — Priority Schedule
# ==========================================================================
with TAB_SCHEDULE:
    st.markdown("#### Leakage Intervention Priority Score")
    active = {k_: v for k_, v in lips_weights.items() if v > 0}
    wtot = sum(active.values())
    wtxt = " · ".join(f"{LIPS_COMPONENTS[k_][0]} {v/wtot*100:.0f}%"
                      for k_, v in sorted(active.items(), key=lambda x: -x[1]))
    st.markdown(T.callout(
        f"LIPS blends six percentile-ranked components into a single 0–100 "
        f"priority score. Percentile rank is used rather than min–max scaling "
        f"because plant size is heavily right-skewed — min–max would let the "
        f"single largest plant compress every other plant into the bottom "
        f"decile. Current weighting: <b>{wtxt}</b>."),
        unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        n_show = min(15, n_plants)
        top = sel.nsmallest(n_show, "lips_rank").sort_values("lips")
        fig = go.Figure(go.Bar(
            x=top.lips, y=top.plant, orientation="h",
            marker=dict(color=top.lips, colorscale=T.SEQ,
                        line=dict(color=T.SURFACE, width=2), showscale=False),
            text=[f"{v:.0f}" for v in top.lips], textposition="outside",
            textfont=dict(size=11, color=T.INK_2),
            customdata=np.stack([top.district, top.nrw_m3, top.nrw_pct,
                                 top.lips_rank], -1),
            hovertemplate=("<b>%{y}</b> · %{customdata[0]}<br>"
                           "LIPS  %{x:.1f}  (rank %{customdata[3]})<br>"
                           "NRW  %{customdata[1]:,.0f} m³<br>"
                           "Loss rate  %{customdata[2]:.1f}%<extra></extra>")))
        fig.update_layout(
            title=f"Top {n_show} plants by intervention priority", height=560,
            bargap=0.3, xaxis=dict(title="LIPS (0–100)",
                                   range=[0, min(105, top.lips.max() * 1.18)]),
            yaxis=dict(title=None, tickfont=dict(size=11)))
        st.plotly_chart(fig, width='stretch', config=PLOT_CFG, theme=None)

    with c2:
        st.markdown("###### What drives each plant's score")
        comp_cols = [f"pr_{c}" for c in active]
        stack = sel.nsmallest(min(15, n_plants), "lips_rank").sort_values("lips")
        fig = go.Figure()
        for i, (col, wt) in enumerate(sorted(active.items(), key=lambda x: -x[1])):
            label = LIPS_COMPONENTS[col][0]
            contrib = stack[f"pr_{col}"] * (wt / wtot)
            fig.add_trace(go.Bar(
                x=contrib, y=stack.plant, orientation="h", name=label,
                marker=dict(color=T.SERIES[i % 8],
                            line=dict(color=T.SURFACE, width=2)),
                customdata=np.stack([stack[f"pr_{col}"],
                                     [wt / wtot * 100] * len(stack)], -1),
                hovertemplate=(f"<b>{label}</b><br>"
                               "Percentile  %{customdata[0]:.0f}<br>"
                               "Weight  %{customdata[1]:.0f}%<br>"
                               "Contribution  %{x:.1f} pts<extra></extra>")))
        fig.update_layout(
            title="LIPS decomposition — contribution to score", height=560,
            barmode="stack", bargap=0.3,
            xaxis=dict(title="Weighted contribution to LIPS"),
            yaxis=dict(title=None, tickfont=dict(size=11)),
            legend=dict(font=dict(size=10.5)))
        st.plotly_chart(fig, width='stretch', config=PLOT_CFG, theme=None)

    st.markdown("---")
    st.markdown("#### Recovery curve — how far a crew programme gets")

    c5, c6 = st.columns([1.3, 1])
    with c6:
        crews = st.slider("Plants a crew programme can reach this year",
                          1, n_plants, min(12, n_plants))
    order_lips = sel.sort_values("lips_rank")
    order_vol = sel.sort_values("volume_rank")
    order_rate = sel.sort_values("rate_rank")
    total_nrw = sel.nrw_m3.sum()

    def curve(df):
        return df.nrw_m3.cumsum() / total_nrw * 100

    x = np.arange(1, n_plants + 1)
    with c5:
        fig = go.Figure()
        for name, df_, col in [("LIPS order", order_lips, T.BLUE),
                               ("Volume order", order_vol, T.ORANGE),
                               ("Rate order", order_rate, T.AQUA)]:
            fig.add_trace(go.Scatter(
                x=x, y=curve(df_), mode="lines", name=name,
                line=dict(color=col, width=2.5),
                hovertemplate=(f"<b>{name}</b><br>First %{{x}} plants<br>"
                               "cover %{y:.1f}% of NRW<extra></extra>")))
        fig.add_vline(x=crews, line=dict(color=T.BASELINE, width=1, dash="dot"))
        fig.update_layout(
            title="Share of total NRW covered, by queue ordering", height=400,
            xaxis=dict(title="Plants visited, in queue order"),
            yaxis=dict(title="% of total NRW covered", ticksuffix="%",
                       range=[0, 102]))
        st.plotly_chart(fig, width='stretch', config=PLOT_CFG, theme=None)

    with c6:
        cov_l = curve(order_lips).iloc[crews - 1]
        cov_v = curve(order_vol).iloc[crews - 1]
        cov_r = curve(order_rate).iloc[crews - 1]
        val_l = order_lips.head(crews).nrw_value_rm.sum()
        val_r = order_rate.head(crews).nrw_value_rm.sum()
        st.markdown(T.tile("LIPS queue", f"{cov_l:.0f}", "% of NRW",
                           f"RM {rm(val_l)} of forgone revenue in scope"),
                    unsafe_allow_html=True)
        st.markdown("")
        st.markdown(T.tile("Pure volume queue", f"{cov_v:.0f}", "% of NRW",
                           "Upper bound — maximises water, ignores condition"),
                    unsafe_allow_html=True)
        st.markdown("")
        st.markdown(T.tile("Rate queue", f"{cov_r:.0f}", "% of NRW",
                           f"RM {rm(val_r)} — what percentage-led targeting reaches"),
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="caption">With {crews} plants in scope, LIPS reaches '
            f'<b>{cov_l:.0f}%</b> of losses against <b>{cov_r:.0f}%</b> for the '
            f'rate-led queue. LIPS sits below the pure volume curve by design: '
            f'it trades a little coverage for asset condition and burst history, '
            f'which is what makes a repair likely to succeed.</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Full intervention schedule")
    sched = sel.sort_values("lips_rank")[
        ["lips_rank", "plant", "district", "area_type", "lips", "nrw_m3",
         "nrw_pct", "physical_loss_m3", "nrw_per_km_m3", "bursts_per_100km",
         "plant_age_yr", "nrw_value_rm", "volume_rank", "rate_rank"]]
    st.dataframe(
        sched, width='stretch', hide_index=True, height=420,
        column_config={
            "lips_rank": st.column_config.NumberColumn("Priority", width="small"),
            "plant": "Plant", "district": "District", "area_type": "Area",
            "lips": st.column_config.ProgressColumn("LIPS", min_value=0,
                                                    max_value=100, format="%.1f"),
            "nrw_m3": st.column_config.NumberColumn("NRW m³", format="%,d"),
            "nrw_pct": st.column_config.NumberColumn("Rate", format="%.1f%%"),
            "physical_loss_m3": st.column_config.NumberColumn("Physical m³", format="%,d"),
            "nrw_per_km_m3": st.column_config.NumberColumn("NRW/km", format="%,d"),
            "bursts_per_100km": st.column_config.NumberColumn("Bursts/100km", format="%.1f"),
            "plant_age_yr": st.column_config.NumberColumn("Age yr", format="%.0f"),
            "nrw_value_rm": st.column_config.NumberColumn("Value RM", format="%,d"),
            "volume_rank": st.column_config.NumberColumn("Vol rank"),
            "rate_rank": st.column_config.NumberColumn("Rate rank")})
    st.download_button("Download schedule (CSV)", sched.to_csv(index=False),
                       f"paip_lips_schedule_{year}.csv", "text/csv")

    if ML_MATCHES_YEAR and blend_crit and crit_weight > 0:
        st.markdown("---")
        st.markdown("###### Effect of folding Criticality into LIPS")
        cmp_ = (sel[["plant", "lips_rank", "lips"]]
                .merge(sel_plain[["plant", "lips_rank"]], on="plant",
                       suffixes=("_blended", "_plain")))
        cmp_["movement"] = cmp_.lips_rank_plain - cmp_.lips_rank_blended
        movers = cmp_.reindex(cmp_.movement.abs().sort_values(ascending=False).index).head(12)
        movers = movers.sort_values("movement")
        fig = go.Figure(go.Bar(
            x=movers.movement, y=movers.plant, orientation="h",
            marker=dict(color=[T.CRITICAL if v > 0 else T.BLUE
                               for v in movers.movement],
                        line=dict(color=T.SURFACE, width=2)),
            text=[f"{'+' if v > 0 else ''}{v}" for v in movers.movement],
            textposition="outside", textfont=dict(size=11, color=T.INK_2),
            customdata=np.stack([movers.lips_rank_plain,
                                 movers.lips_rank_blended], -1),
            hovertemplate=("<b>%{y}</b><br>Without criticality  rank "
                           "%{customdata[0]}<br>With criticality  rank "
                           "%{customdata[1]}<extra></extra>")))
        fig.add_vline(x=0, line=dict(color=T.BASELINE, width=1.5))
        fig.update_layout(
            title="Largest rank movements when criticality is included",
            height=400, bargap=0.3,
            xaxis=dict(title="Places gained (positive) or lost"),
            yaxis=dict(title=None, tickfont=dict(size=11)))
        st.plotly_chart(fig, width='stretch', config=PLOT_CFG, theme=None)
        st.markdown(
            '<div class="caption">Red bars are plants the model promotes — they '
            'carry loss the deterministic score alone does not weigh. Comparing '
            'the two orderings is the point of keeping the blend optional.</div>',
            unsafe_allow_html=True)


# ==========================================================================
# TAB 4 — Early Warning (model-based)
# ==========================================================================

        

    


# ==========================================================================
# TAB 5 — Loss Composition
# ==========================================================================
with TAB_COMPOSITION:
    st.markdown("#### Physical leakage versus commercial loss")
    st.markdown(T.callout(
        "Physical (real) losses are water escaping the network — pipe repair "
        "and pressure management recover them. Commercial (apparent) losses are "
        "water delivered but not billed — meter under-registration, "
        "unauthorised connections, billing lag — and they need metering and "
        "enforcement instead. The two demand entirely different interventions, "
        "so the split determines <i>which</i> crew to send, not just where."),
        unsafe_allow_html=True)

    c1, c2 = st.columns([1.3, 1])
    with c1:
        n_show = min(16, n_plants)
        comp = sel.nlargest(n_show, "nrw_m3").sort_values("nrw_m3")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comp.physical_loss_m3, y=comp.plant, orientation="h",
            name="Physical leakage", marker=dict(color=T.BLUE,
                                                 line=dict(color=T.SURFACE, width=2)),
            customdata=comp.physical_share_pct,
            hovertemplate=("<b>%{y}</b><br>Physical  %{x:,.0f} m³ "
                           "(%{customdata:.0f}% of loss)<extra></extra>")))
        fig.add_trace(go.Bar(
            x=comp.commercial_loss_m3, y=comp.plant, orientation="h",
            name="Commercial loss", marker=dict(color=T.ORANGE,
                                                line=dict(color=T.SURFACE, width=2)),
            customdata=100 - comp.physical_share_pct,
            hovertemplate=("<b>%{y}</b><br>Commercial  %{x:,.0f} m³ "
                           "(%{customdata:.0f}% of loss)<extra></extra>")))
        fig.update_layout(
            title=f"Loss composition, {n_show} largest-loss plants", height=560,
            barmode="stack", bargap=0.3,
            xaxis=dict(title="Volume lost (m³)"),
            yaxis=dict(title=None, tickfont=dict(size=11)))
        st.plotly_chart(fig, width='stretch', config=PLOT_CFG, theme=None)

 
    


# ==========================================================================
# TAB 5 — Financial Impact
# ==========================================================================


    


# ==========================================================================
# TAB 6 — Plant Profile
# ==========================================================================
with TAB_PLANT:
    st.markdown("#### Plant profile")
    plant_list = sel.sort_values("lips_rank").plant.tolist()
    c0a, c0b = st.columns([1, 2])
    with c0a:
        plant = st.selectbox("Plant", plant_list, index=0)
    p = sel[sel.plant == plant].iloc[0]
    pm = msel[msel.plant == plant].sort_values("date")
    hist = monthly[monthly.plant == plant].sort_values("date")

    with c0b:
        st.markdown(
            f'<div class="caption" style="padding-top:30px">'
            f'<b>{plant}</b> · {p.district} · {p.area_type} · '
            f'{p.plant_age_yr:.0f}-year-old plant · '
            f'{p.pipe_length_km:,.0f} km of main · '
            f'{p.customer_accounts:,.0f} connections · '
            f'{p.population_served:,.0f} people served</div>',
            unsafe_allow_html=True)

    k = st.columns(5)
    k[0].markdown(T.tile("LIPS", f"{p.lips:.1f}", f"rank {p.lips_rank}",
                         f"of {n_plants} plants in the current selection"),
                  unsafe_allow_html=True)
    k[1].markdown(T.tile("Loss rate", f"{p.nrw_pct:.1f}", "%",
                         f"Rank {p.rate_rank} · system average {sys_pct:.1f}%"),
                  unsafe_allow_html=True)
    k[2].markdown(T.tile("Water lost", m3(p.nrw_m3), "m³",
                         f"Rank {p.volume_rank} · {p.nrw_m3/tot_nrw*100:.1f}% of selection total"),
                  unsafe_allow_html=True)
    k[3].markdown(T.tile("Physical share", f"{p.physical_share_pct:.0f}", "%",
                         f"{m3(p.physical_loss_m3)} m³ addressable by repair"),
                  unsafe_allow_html=True)
    k[4].markdown(T.tile("Value at stake", "RM " + rm(p.nrw_value_rm), "",
                         f"{p.bursts_per_100km:.1f} bursts per 100 km recorded"),
                  unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hist.date, y=hist.physical_loss_m3, name="Physical leakage",
            marker=dict(color=T.BLUE, line=dict(color=T.SURFACE, width=1)),
            hovertemplate="%{x|%b %Y}<br>Physical  %{y:,.0f} m³<extra></extra>"))
        fig.add_trace(go.Bar(
            x=hist.date, y=hist.commercial_loss_m3, name="Commercial loss",
            marker=dict(color=T.ORANGE, line=dict(color=T.SURFACE, width=1)),
            hovertemplate="%{x|%b %Y}<br>Commercial  %{y:,.0f} m³<extra></extra>"))
        fig.update_layout(
            title=f"{plant} — monthly loss volume, {YEAR_SPAN}", height=380,
            barmode="stack", bargap=0.15,
            xaxis=dict(title=None), yaxis=dict(title="Volume lost (m³)"))
        st.plotly_chart(fig, width='stretch', config=PLOT_CFG, theme=None)

    with c2:
        peer = sel[(sel.area_type == p.area_type)]
        metrics = [("Loss rate %", "nrw_pct"), ("NRW per km", "nrw_per_km_m3"),
                   ("Bursts /100km", "bursts_per_100km"),
                   ("Plant age", "plant_age_yr"), ("Meter age", "meter_age_yr")]
        labels, pvals, medians = [], [], []
        for label, col in metrics:
            med = peer[col].median()
            if med and not np.isnan(med):
                labels.append(label)
                pvals.append(p[col] / med * 100)
                medians.append(med)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pvals, y=labels, orientation="h",
            marker=dict(color=[T.CRITICAL if v > 130 else
                               T.BLUE if v > 70 else T.GOOD for v in pvals],
                        line=dict(color=T.SURFACE, width=2)),
            text=[f"{v:.0f}" for v in pvals], textposition="outside",
            textfont=dict(size=11, color=T.INK_2),
            customdata=medians,
            hovertemplate=("<b>%{y}</b><br>%{x:.0f}% of peer median<br>"
                           "Peer median  %{customdata:,.1f}<extra></extra>")))
        fig.add_vline(x=100, line=dict(color=T.BASELINE, width=1.5, dash="dash"),
                      annotation_text="peer median",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10.5, color=T.MUTED))
        fig.update_layout(
            title=f"Against {p.area_type} peers (n={len(peer)}), median = 100",
            height=380, bargap=0.35,
            xaxis=dict(title="% of peer median",
                       range=[0, max(pvals) * 1.25 if pvals else 200]),
            yaxis=dict(title=None))
        st.plotly_chart(fig, width='stretch', config=PLOT_CFG, theme=None)

    if HAS_ML and not pd.isna(p.get("criticality", np.nan)):
        st.markdown("---")
        st.markdown("###### What the model sees at this plant")
        pm_ml = ml_monthly[ml_monthly.plant == plant].sort_values("date")
        c7, c8 = st.columns([1.5, 1])
        with c7:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pm_ml.date, y=pm_ml.predicted_nrw_pct, mode="lines",
                name="Expected from plant characteristics",
                line=dict(color=T.ORANGE, width=2, dash="dash"),
                hovertemplate="%{x|%b %Y}<br>Expected  %{y:.1f}%<extra></extra>"))
            fig.add_trace(go.Scatter(
                x=pm_ml.date, y=pm_ml.nrw_pct, mode="lines",
                name="Actual", line=dict(color=T.BLUE, width=2.5),
                hovertemplate="%{x|%b %Y}<br>Actual  %{y:.1f}%<extra></extra>"))
            an = pm_ml[pm_ml.is_anomaly.fillna(False).astype(bool)]
            if len(an):
                fig.add_trace(go.Scatter(
                    x=an.date, y=an.nrw_pct, mode="markers",
                    name="Anomalous month",
                    marker=dict(size=13, color=T.CRITICAL, symbol="circle-open",
                                line=dict(width=2.5)),
                    hovertemplate=("%{x|%b %Y}<br>Anomaly · robust z "
                                   "%{customdata:.1f}<extra></extra>"),
                    customdata=an.robust_z))
            fig.update_layout(
                title=f"{plant} — actual loss against model expectation",
                height=350, xaxis=dict(title=None),
                yaxis=dict(title="NRW (%)", ticksuffix="%"))
            st.plotly_chart(fig, width='stretch', config=PLOT_CFG, theme=None)
            gap = float(p.unexplained_pp)
            verdict = ("above" if gap > 0 else "below")
            st.markdown(
                f'<div class="caption">This plant runs <b>{abs(gap):.1f} pp '
                f'{verdict}</b> what its network characteristics predict — '
                f'{abs(float(p.unexplained_m3)):,.0f} m³ a year '
                f'{"unaccounted for" if gap > 0 else "better than expected"}. '
                f'Pattern: <b>{p.archetype}</b>.</div>',
                unsafe_allow_html=True)
        with c8:
            sig = pd.DataFrame({
                "Signal": ["Criticality rank", "Unexplained loss",
                           "Trend (36 mo)", "Recent trend (12 mo)",
                           "Step change (last 6 mo)", "Anomalous months",
                           "Month-to-month volatility"],
                "Value": [f"{int(p.criticality_rank)} of {n_plants}",
                          f"{p.unexplained_pp:+.1f} pp",
                          f"{p.trend_pp_yr:+.2f} pp/yr (p={p.trend_p:.3f})",
                          f"{p.trend_recent_pp_yr:+.2f} pp/yr",
                          f"{p.step_shift_pp:+.2f} pp (p={p.step_p:.3f})",
                          f"{int(p.anomaly_months)}",
                          f"{p.volatility_pp:.2f} pp"]})
            st.dataframe(sig, width='stretch', hide_index=True,
                         height=300)
            st.markdown('<div class="caption">A negative trend means '
                        'improving. p-values above 0.05 mean the movement is '
                        'not distinguishable from noise.</div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist.date, y=hist.nrw_pct, mode="lines+markers", name="Loss rate",
            line=dict(color=T.BLUE, width=2),
            marker=dict(size=5, color=T.BLUE),
            hovertemplate="%{x|%b %Y}<br>Loss rate  %{y:.1f}%<extra></extra>"))
        fig.add_hline(y=T.POLICY_TARGET_PCT,
                      line=dict(color=T.GOOD, width=1.2, dash="dash"),
                      annotation_text="25% target",
                      annotation_font=dict(size=10.5, color=T.SUCCESS_TEXT))
        fig.update_layout(title="Loss rate history", height=320,
                          showlegend=False, xaxis=dict(title=None),
                          yaxis=dict(title="NRW (%)", ticksuffix="%"))
        st.plotly_chart(fig, width='stretch', config=PLOT_CFG, theme=None)

    with c4:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hist.date, y=hist.pipe_bursts, name="Bursts",
            marker=dict(color=T.ORANGE, line=dict(color=T.SURFACE, width=1)),
            hovertemplate="%{x|%b %Y}<br>%{y:.0f} bursts<extra></extra>"))
        fig.update_layout(title="Recorded pipe bursts", height=320,
                          showlegend=False, bargap=0.15,
                          xaxis=dict(title=None),
                          yaxis=dict(title="Bursts in month"))
        st.plotly_chart(fig, width='stretch', config=PLOT_CFG, theme=None)

    with st.expander(f"Monthly records for {plant}"):
        cols = ["date", "production_m3", "billed_m3", "nrw_m3", "nrw_pct",
                "physical_loss_m3", "commercial_loss_m3", "pipe_bursts",
                "complaints", "pressure_bar", "rainfall_mm", "nrw_value_rm"]
        st.dataframe(hist[cols], width='stretch', hide_index=True,
                     column_config={
                         "date": st.column_config.DateColumn("Month", format="MMM YYYY"),
                         "production_m3": st.column_config.NumberColumn("Production m³", format="%,d"),
                         "billed_m3": st.column_config.NumberColumn("Billed m³", format="%,d"),
                         "nrw_m3": st.column_config.NumberColumn("NRW m³", format="%,d"),
                         "nrw_pct": st.column_config.NumberColumn("Rate", format="%.1f%%"),
                         "physical_loss_m3": st.column_config.NumberColumn("Physical m³", format="%,d"),
                         "commercial_loss_m3": st.column_config.NumberColumn("Commercial m³", format="%,d"),
                         "pipe_bursts": st.column_config.NumberColumn("Bursts"),
                         "complaints": st.column_config.NumberColumn("Complaints"),
                         "pressure_bar": st.column_config.NumberColumn("Pressure bar", format="%.2f"),
                         "rainfall_mm": st.column_config.NumberColumn("Rain mm", format="%.0f"),
                         "nrw_value_rm": st.column_config.NumberColumn("Value RM", format="%,d")})


# ==========================================================================
# TAB 7 — Method & Data Quality
# ==========================================================================





# ==========================================================================
# TAB 9 — Data Management
# ==========================================================================

    
