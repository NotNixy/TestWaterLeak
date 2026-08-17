"""Shared visual language for the PAIP NRW dashboard.

Light and dark are both *selected*, not flipped. The dark column is the same
eight hues re-stepped for the dark surface, each validated against that surface
rather than derived by inverting the light values.

Palette values follow a validated categorical order (adjacent-pair CVD deltaE >= 8,
normal-vision deltaE >= 15 in both modes). Slots are assigned in fixed order and
never cycled; a 9th series folds into "Other" rather than generating a new hue.
Scatter, bubble and cluster forms are capped at the first three slots, which are
the ones that clear the all-pairs floors.
"""

import plotly.graph_objects as go
import plotly.io as pio

# ---- Categorical slots, both modes ---------------------------------------
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]

# ---- Sequential ramp (single hue, light -> dark) -------------------------
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
# On the dark surface the ramp runs the other way: the step nearest the surface
# must still separate from it, so the light end leads.
SEQ_BLUE_DARK = list(reversed(SEQ_BLUE[:-2]))

# ---- Status (reserved; always paired with an icon or label) --------------
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"

# ---- Chrome and ink, per mode -------------------------------------------
PALETTES = {
    "light": dict(
        series=SERIES_LIGHT, seq=SEQ_BLUE,
        surface="#fcfcfb", page="#f9f9f7",
        ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", baseline="#c3c2b7",
        success_text="#006300", neutral="#c9c8c2",
        border="rgba(11,11,11,0.10)", hover_bg="#ffffff",
        tile_wash="rgba(42,120,214,0.10)",
    ),
    "dark": dict(
        series=SERIES_DARK, seq=SEQ_BLUE_DARK,
        surface="#1a1a19", page="#0d0d0d",
        ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", baseline="#383835",
        success_text="#0ca30c", neutral="#55554f",
        border="rgba(255,255,255,0.10)", hover_bg="#252523",
        tile_wash="rgba(57,135,229,0.16)",
    ),
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

NATIONAL_NRW_PCT = 35.0
POLICY_TARGET_PCT = 25.0


class Theme:
    """Resolved palette for one mode. Charts are written against roles, so the
    whole dashboard re-colours by swapping this object."""

    def __init__(self, mode: str = "light"):
        self.mode = mode if mode in PALETTES else "light"
        p = PALETTES[self.mode]
        self.SERIES = p["series"]
        (self.BLUE, self.ORANGE, self.AQUA, self.YELLOW,
         self.MAGENTA, self.GREEN, self.VIOLET, self.RED) = p["series"]
        self.SEQ = p["seq"]
        self.SURFACE = p["surface"]
        self.PAGE = p["page"]
        self.INK = p["ink"]
        self.INK_2 = p["ink2"]
        self.MUTED = p["muted"]
        self.GRID = p["grid"]
        self.BASELINE = p["baseline"]
        self.SUCCESS_TEXT = p["success_text"]
        self.NEUTRAL = p["neutral"]
        self.BORDER = p["border"]
        self.HOVER_BG = p["hover_bg"]
        self.TILE_WASH = p["tile_wash"]
        self.GOOD, self.WARNING = GOOD, WARNING
        self.SERIOUS, self.CRITICAL = SERIOUS, CRITICAL
        self.POLICY_TARGET_PCT = POLICY_TARGET_PCT
        self.NATIONAL_NRW_PCT = NATIONAL_NRW_PCT
        self.FONT = FONT
        self.template = f"paip_{self.mode}"
        self._install()

    def _install(self):
        tpl = go.layout.Template()
        tpl.layout = go.Layout(
            font=dict(family=FONT, size=13, color=self.INK_2),
            title=dict(font=dict(size=15, color=self.INK), x=0,
                       xanchor="left", y=0.97, yanchor="top"),
            paper_bgcolor=self.SURFACE,
            plot_bgcolor=self.SURFACE,
            colorway=self.SERIES,
            margin=dict(l=10, r=10, t=84, b=10),
            hoverlabel=dict(bgcolor=self.HOVER_BG, bordercolor=self.BASELINE,
                            font=dict(family=FONT, size=12, color=self.INK)),
            xaxis=dict(gridcolor=self.GRID, gridwidth=1, zeroline=False,
                       linecolor=self.BASELINE, linewidth=1, ticks="outside",
                       ticklen=4, tickcolor=self.BASELINE,
                       tickfont=dict(color=self.MUTED, size=11),
                       # Long plant names on horizontal bar charts were being
                       # clipped by the 10px margin; automargin grows it instead.
                       automargin=True,
                       title=dict(font=dict(color=self.INK_2, size=12))),
            yaxis=dict(gridcolor=self.GRID, gridwidth=1, zeroline=False,
                       linecolor=self.BASELINE, linewidth=1, ticks="outside",
                       ticklen=4, tickcolor=self.BASELINE,
                       tickfont=dict(color=self.MUTED, size=11),
                       # Long plant names on horizontal bar charts were being
                       # clipped by the 10px margin; automargin grows it instead.
                       automargin=True,
                       title=dict(font=dict(color=self.INK_2, size=12))),
            legend=dict(orientation="h", yanchor="bottom", y=1.015,
                        xanchor="left", x=0,
                        font=dict(size=12, color=self.INK_2),
                        bgcolor="rgba(0,0,0,0)", traceorder="normal"),
        )
        pio.templates[self.template] = tpl
        pio.templates.default = self.template

    # -- HTML helpers ------------------------------------------------------
    def tile(self, label, value, unit="", sub=""):
        u = f'<span class="tile-unit"> {unit}</span>' if unit else ""
        s = f'<div class="tile-sub">{sub}</div>' if sub else ""
        return (f'<div class="tile"><div class="tile-label">{label}</div>'
                f'<div class="tile-value">{value}{u}</div>{s}</div>')

    def callout(self, text, kind=""):
        cls = {"warn": " callout-warn", "crit": " callout-crit",
               "good": " callout-good"}.get(kind, "")
        return f'<div class="callout{cls}">{text}</div>'

    @property
    def css(self):
        p = self
        return f"""
<style>
  .stApp {{ background: {p.PAGE}; }}
  html, body, [class*="css"] {{ font-family: {FONT}; }}
  .block-container {{ padding-top: 2.2rem; max-width: 1500px; }}

  h1, h2, h3, h4, h5, h6 {{ color: {p.INK}; letter-spacing: -0.01em; }}
  .stApp, .stMarkdown, p, span, label, li {{ color: {p.INK_2}; }}

  .tile {{
    background: {p.SURFACE}; border: 1px solid {p.BORDER};
    border-radius: 10px; padding: 16px 18px 14px 18px; height: 100%;
  }}
  .tile-label {{
    font-size: 11.5px; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; color: {p.MUTED}; margin-bottom: 6px;
  }}
  .tile-value {{ font-size: 30px; font-weight: 650; color: {p.INK}; line-height: 1.1; }}
  .tile-unit {{ font-size: 15px; font-weight: 500; color: {p.INK_2}; }}
  .tile-sub {{ font-size: 12px; color: {p.INK_2}; margin-top: 6px; line-height: 1.45; }}
  .tile-delta-good {{ color: {p.SUCCESS_TEXT}; font-weight: 600; }}
  .tile-delta-bad {{ color: {p.CRITICAL}; font-weight: 600; }}

  .callout {{
    background: {p.SURFACE}; border: 1px solid {p.BORDER};
    border-left: 3px solid {p.BLUE}; border-radius: 8px;
    padding: 14px 18px; margin: 6px 0 18px 0;
    font-size: 13.5px; color: {p.INK_2}; line-height: 1.6;
  }}
  .callout-warn {{ border-left-color: {WARNING}; }}
  .callout-crit {{ border-left-color: {CRITICAL}; }}
  .callout-good {{ border-left-color: {GOOD}; }}
  .callout b {{ color: {p.INK}; }}

  .caption {{ font-size: 12px; color: {p.MUTED}; line-height: 1.55;
              margin: -6px 0 16px 0; }}
  .caption b {{ color: {p.INK_2}; }}

  .stTabs [role="tablist"] {{ gap: 2px; border-bottom: 1px solid {p.GRID}; }}
  .stTabs [data-testid="stTab"] {{ height: 42px; padding: 0 16px;
                                   background: transparent; }}
  .stTabs [data-testid="stTab"] p {{ font-size: 13.5px; font-weight: 500;
                                     color: {p.MUTED}; margin: 0; }}
  .stTabs [data-testid="stTab"][aria-selected="true"] p {{
    color: {p.INK}; font-weight: 600; }}

  section[data-testid="stSidebar"] {{ background: {p.SURFACE};
    border-right: 1px solid {p.GRID}; }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}

  div[data-testid="stDataFrame"] {{ border-radius: 8px; }}
  hr {{ border-color: {p.GRID}; margin: 1.4rem 0; }}

  /* Accent override ---------------------------------------------------
     config.toml deliberately sets no [theme], because setting ANY theme key
     makes Streamlit resolve a concrete base and forces its widget chrome and
     built-in Plotly theme to light even when the OS is dark. The cost is that
     Streamlit keeps its default red accent (#ff4b4b), which collides with the
     status palette's "critical" and reads as an alert on every chip and
     slider. These rules re-point the accent at the palette blue using stable
     data-testid / react-aria hooks rather than build-specific emotion hashes. */
     The :not([data-testid]) guards matter: without them these rules also paint
     the label's stMarkdownContainer and the slider's tick bar, which turns the
     option TEXT into a blue block instead of tinting the control. */
  [data-testid="stRadioOption"] > div > div > div:not([data-testid]) {{
    background-color: {p.BLUE} !important; }}
  [data-testid="stMultiSelectTagsContainer"] span[role="group"] > span {{
    background-color: {p.BLUE} !important; }}
  [data-testid="stSlider"] div[role="group"] > div:not([data-testid])
    > div:not([data-testid]) {{ background-color: {p.BLUE} !important; }}
  [data-testid="stSliderThumbValue"] {{
    color: {p.BLUE} !important; border-color: {p.BLUE} !important; }}
  .react-aria-SelectionIndicator {{
    background-color: {p.BLUE} !important; color: {p.BLUE} !important;
    border-color: {p.BLUE} !important; }}
  [data-testid="stTab"] {{ color: {p.MUTED} !important; }}
  [data-testid="stCheckbox"] svg {{ color: {p.BLUE}; }}
</style>
"""


def resolve_mode(preference: str, detected: str) -> str:
    """`preference` is the sidebar override; `detected` comes from the browser
    or OS via st.context.theme. Auto follows the system."""
    if preference == "Light":
        return "light"
    if preference == "Dark":
        return "dark"
    return detected if detected in PALETTES else "light"
