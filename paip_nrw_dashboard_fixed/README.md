# PAIP Non-Revenue Water — Intervention Priority Dashboard

A Streamlit dashboard that turns Pengurusan Air Pahang Berhad's published monthly
production and billing figures into a ranked, volume-weighted repair schedule.

The operational question is not *how large are the losses* — PAIP already knows —
but *which plants to fix first*. NRW is published as a percentage because a
percentage makes plants comparable; but a percentage measures efficiency, not
recoverable water. This dashboard shows both, and quantifies how far apart they
are.

---

## Running it

```bash
pip install -r requirements.txt
# put the PAIP export in data/raw/ (any filename)
python refresh.py           # clean -> train -> verify, one command
streamlit run app.py        # opens on http://localhost:8501
```

Verify or test at any time:

```bash
python refresh.py --dry-run # validate the input, write nothing
python verify.py            # 50 independent checks against the raw CSV
python test_refresh.py      # 25 checks that next year's data will load
python shoot.py             # headless render test, both colour modes
```

**Light and dark mode.** The dashboard follows your operating system setting
automatically. The sidebar "Appearance" control overrides it per session. Both
palettes are separately validated against their own surface rather than one
being an inversion of the other.

---

## The headline finding

Ranking the same 74 plants by loss **rate** and by loss **volume** produces two
almost unrelated queues:

| Measure | Value |
|---|---|
| Spearman ρ between rate and volume | **−0.54** (negative — they rank plants *oppositely*) |
| Kendall τ between the two rank orders | −0.35 |
| Plants shared by the two top-10 queues | **0 of 10** |
| Water in the top-10-by-volume queue | 140.2M m³ |
| Water in the top-10-by-rate queue | 8.9M m³ |
| Ratio | **15.8× more water for the same ten crew deployments** |

Losses are also highly concentrated: **65% of all non-revenue water sits in 10 of
74 plants**. Monthly loss rates move within a 0.9 pp band across 2025 with no
wet/dry-season signal, which is the signature of continuous physical leakage
rather than seasonal demand or intermittent billing error — and continuous
leakage is what repair work recovers.

System figures for 2025: **32.1%** loss rate, **216.5M m³** lost, **RM 268.5M**
in forgone revenue at the published average tariff, **68%** of it physical.

---

## The LIPS score

The Leakage Intervention Priority Score blends six components, each converted to
a 0–100 percentile rank within the filtered cohort, then weighted:

| Component | Default weight | Why it is in the score |
|---|---|---|
| Physical loss volume | 35% | m³ of leakage a repair physically recovers |
| Leak concentration (NRW per km) | 15% | water recovered per crew-day |
| Loss rate | 13% | the efficiency signal, kept in the frame |
| Burst frequency per 100 km | 12% | observed network deterioration |
| Asset condition | 15% | plant age (60%) + meter age (40%) |
| Repairable share | 10% | share of loss that is physical, not commercial |

**Percentile rank, not min–max.** Plant size is heavily right-skewed — production
spans from 31.6k to 7.2M m³ per month. Min–max scaling would let the single
largest plant compress every other plant into the bottom decile, making the score
a proxy for "is this the biggest plant".

**Ties are broken deterministically** on physical loss volume, so the schedule is
a strict order — two plants can never both be "priority 31". (Real ties occur in
both 2024 and 2025; an earlier `rank(method="min")` implementation produced
duplicate priority numbers and was caught by `verify.py`.)

**The weights are a management judgement, not an estimated parameter.** The
sidebar sliders exist so the ranking can be stress-tested against alternative
priorities — set a weight to 0 to drop a component entirely.

---

## The early-warning models

LIPS asks *where is the most water*. The Early Warning tab asks *where is
something wrong* — which plants lose more than their physical characteristics
can account for.

**There are no ground-truth labels in this dataset** — no repair records, no
inspections, no confirmed defects. Nothing here is a trained failure classifier
and it is not presented as one. What is modelled is *expected loss given plant
characteristics*; a large positive residual means "this plant is unusual for its
type", which is a lead to investigate, not a diagnosis.

### 1. Unexplained loss

A regression model predicts monthly NRW% from 26 asset, network, operational and
environmental features. Three candidates compete; the winner on grouped CV is
used.

| Model | R² (unseen plants) | MAE pp | R² (forward in time) |
|---|---|---|---|
| Ridge regression | **0.451** ✓ selected | **5.16** | 0.613 |
| Gradient boosting | 0.384 | 5.35 | 0.652 |
| Mean baseline | −0.036 | 7.31 | −0.040 |

Two design decisions carry this result:

**Validation is grouped by plant.** Static characteristics (pipe length, age,
capacity) are near-constant within a plant, so a random split would let the model
memorise each plant's own loss level and the residual would collapse toward zero.
`GroupKFold` forces every prediction to come from a model that has never seen the
plant it is scoring. `verify.py` asserts zero plant overlap between train and
test folds — the claim is tested, not asserted.

**The simpler model won, and it was selected on that basis.** An unconstrained
booster (depth 6, 400 iterations) scored just 0.21 on unseen plants: with only 74
plant groups it spends its capacity on plant-specific structure that does not
transfer. Heavy regularisation lifted it to 0.38, still short of ridge. Note the
reversal in the last column — gradient boosting looks *better* forward in time
(0.652) precisely because the same plants appear on both sides of that split and
it can memorise them. That number is optimistic and is **not** what the residual
rests on.

**Leakage control is the central safeguard.** NRW is production minus billed
volume, so any feature carrying billed volume encodes the target algebraically.
Two source columns are exactly this trap and are excluded:

| Column | Actually equals | Verdict |
|---|---|---|
| `consumption_per_capita_l_day` | billed ÷ population ÷ days | leaks — excluded |
| `revenue_per_account_rm` | billed × tariff ÷ accounts | leaks — excluded |
| `capacity_utilisation_pct` | production ÷ (capacity × days) | safe — retained |
| `energy_intensity_kwh_m3` | kWh ÷ production | safe — retained |

Each was confirmed numerically, not assumed. The exclusion list is asserted at
runtime and re-derived independently in `verify.py`.

### 2. Sudden deterioration — a null result

Robust per-plant z-scores (median/MAD), a global Isolation Forest, and a Welch
step-change test comparing the last 6 months against prior history.

**These detectors found nothing, and the dashboard says so.** Across 74 plants
and 36 months: **0** plants with a statistically significant worsening trend,
**0** with a significant step increase, and **1** month-level anomaly in 2,664
records. The estate is improving uniformly at a median **1.06 pp/year**.

That is a genuine finding, not a failure. It says PAIP's losses are *chronic and
structural*, not the result of sudden failures — which is why criticality here is
driven almost entirely by unexplained loss. Because every plant is improving, the
trend component measures *improving more slowly than peers* rather than outright
worsening; an absolute test would flag nobody and the component would be a
constant.

### 3. Criticality Index

Percentile-rank blend: unexplained loss 40%, sudden deterioration 30%, relative
trend 30%. Ties break on unexplained volume.

Criticality is **not** a restatement of LIPS — `verify.py` checks this. Spearman
ρ against LIPS is 0.43 and against NRW volume 0.15, so it carries genuinely
different information. Top of the 2025 ranking:

| # | Plant | District | Criticality | Unexplained |
|---|---|---|---|---|
| 1 | SG BILUT | RAUB | 84.3 | +2.4 pp |
| 2 | BERA KOMPLEKS | BERA | 82.6 | +9.6 pp |
| 3 | KECHAU | LIPIS | 80.3 | +8.2 pp (534k m³) |
| 4 | SG BERA (KEPAYANG) | BERA | 80.0 | +14.2 pp |
| 5 | PADANG PIOL | JERANTUT | 79.7 | +6.9 pp (562k m³) |

Across the estate, **7.4M m³** — about 3% of all NRW — sits above what the model
predicts from plant characteristics.

### 4. Failure archetypes

KMeans over the loss signature, named from the two features whose centroid
deviates most from the estate, each mapped to the intervention it implies.

**Separation is weak** (silhouette 0.20 at the best k). The estate varies
continuously rather than falling into discrete failure types. The tab shows the
silhouette curve for k = 2…6 and lets you change k, precisely so this is visible
rather than hidden behind a confident-looking label. Treat the archetypes as a
communication aid, not evidence of real categories.

### Optional blend with LIPS

Off by default. The sidebar can fold criticality in as a seventh LIPS component,
and the Priority Schedule then shows which plants the model promotes or demotes.
Keeping it optional is the point: the deterministic score and the model-based one
stay comparable.

---

## Adding next year's data

**Nothing is hard-coded to a year.** Every year label, heading, split and model
focus is derived from what is in `data/raw/`. Dropping 2026 in and running
`python refresh.py` is the whole procedure — no code change.

### Two arrival patterns, auto-detected

| Pattern | What you do |
|---|---|
| **Full replacement** | PAIP republishes one workbook covering every year — replace the file in `data/raw/` |
| **Per-year append** | Drop `paip_2026.csv` alongside the existing file; everything in the folder is concatenated |

Records are de-duplicated on plant and month keeping the **last** occurrence, so
a corrected re-issue of an old year supersedes the original instead of
double-counting it.

### The refresh command

```bash
python refresh.py --dry-run              # validate only, write nothing
python refresh.py                        # rebuild from data/raw/
python refresh.py --add ~/paip_2026.csv  # copy the file in, then rebuild
```

Stages run in dependency order: **clean → train → verify**. Clean failing stops
everything, because training on a malformed extract produces confident nonsense.
Verify failing means the rebuilt artefacts disagree with the raw data — do not
publish that build. Existing artefacts are backed up to `data/_backup/` first,
and restored automatically if the clean stage fails.

There is also a **Data Management** tab in the dashboard: upload a CSV to see
the full validation report before committing to a rebuild.

### Partial years are kept, flagged and annualised

A year still in progress is not hidden and not silently mixed with complete
years. It is labelled everywhere (`2026 · 7 of 12 months`), drawn in the amber
status colour on year-comparison charts with a footnote, and its volumes are
annualised wherever years are compared. Rates are ratios and need no adjustment.
Both raw and annualised volumes are stored, so the UI shows actuals while
charting comparables.

### What the loader rejects

Structural problems raise and stop the build; everything else is a warning and
the build proceeds.

| Check | Severity |
|---|---|
| Required columns present (named as they appear in *your* file) | Error |
| Dates parse unambiguously — day-first tried first, matching PAIP | Error |
| Production is positive | Error |
| Each plant maps to exactly one district | Error |
| NRW = production − billed | Warning — published figure kept |
| Negative NRW | Warning — retained and flagged, never corrected |
| Duplicate plant-months | Warning — last wins, treated as a correction |
| Year gaps | Warning |
| New or vanished plants | Note — a vanished plant is usually a rename |
| Incomplete years | Note |

### Proof it works

`test_refresh.py` fabricates next year's data and runs nine scenarios end to
end — full year appended, partial year, full replacement, corrected re-issue,
and five deliberately malformed files that **must** be rejected. All 25 checks
pass. The synthetic year is built to be internally consistent, matching PAIP's
own per-column decimal precision, so it exercises the pipeline rather than
tripping identity checks for the wrong reason.

---

## Tabs

| Tab | What it answers |
|---|---|
| **Overview** | Where the water goes; how concentrated losses are; seasonality; three-year trajectory against the 25% policy target and 35% national average |
| **Rate vs Volume** | The core argument — two measures, two different repair queues, quantified |
| **Priority Schedule** | LIPS ranking, per-plant score decomposition, recovery curve by queue ordering, full downloadable schedule |
| **Early Warning** | Model-based criticality, actual-vs-expected loss, validation evidence, feature importance, failure archetypes |
| **Data Management** | Coverage by year, upload-and-validate a new export, refresh procedure |
| **Loss Composition** | Physical vs commercial split; whether leakage tracks bursts, age, pressure; loss normalised per km and per connection |
| **Financial Impact** | Forgone revenue vs sunk production cost; value at stake against priority; cost recovery |
| **Plant Profile** | Per-plant drill-down: 36-month history, peer comparison, burst record |
| **Method & Data Quality** | Identity checks, missing values, LIPS specification, limitations, plant crosswalk |

---

## Files

```
app.py                 the dashboard
theme.py               light + dark palettes, chart template, CSS
dataloader.py          ingestion, schema validation, de-duplication
prepare_data.py        cleaning and feature engineering
train_models.py        expected-loss model, anomaly detection, clustering
refresh.py             one-command rebuild: clean -> train -> verify
verify.py              50 independent checks against the raw CSV
test_refresh.py        25 checks that next year's data will load correctly
shoot.py               headless render test — every tab, both colour modes
requirements.txt
.streamlit/config.toml
data/
  raw/                 PUT THE PAIP EXPORT HERE
  year_coverage.csv    months observed per year, annualisation factors
  nrw_plant_month.csv  2,664 tidy plant-month records
  nrw_plant_year.csv   222 plant-year rows with LIPS
  plant_crosswalk.csv  74 plants → district / region / area type
  data_quality.csv     identity check results
  missing_values.csv
  ml_plant.csv         per-plant criticality, residuals, archetypes
  ml_monthly.csv       per-plant-month predictions and anomaly flags
  model_metrics.json   validation table, importances, cluster profiles
shots/                 screenshots of every tab, light and dark
```

---

## Data quality

Every published figure was recomputed from its components. Nothing was silently
corrected.

| Check | Max deviation |
|---|---|
| `billed = domestic + commercial + industrial` | 0 |
| `nrw = production − billed` | 0 |
| `nrw_pct = nrw / production` | 0.05 pp (source rounding) |
| `physical + commercial loss = nrw` | 1 m³ (source rounding) |
| `billed_revenue = billed × tariff` | RM 0.005 |
| `opex = energy + chemical + maintenance` | ~0 |
| Negative NRW records | **0** |
| Months where billed > production | **0** |

Missing values: `pressure_bar` (58 rows, 2.18%) and `raw_turbidity_ntu` (89 rows,
3.34%). Both are sensor-derived, neither is a LIPS input, and both are excluded
pairwise from the correlations that use them. No imputation was performed.

---

## Deviation from the original proposal

The dataset supplied differs materially from the one the proposal described. The
dashboard was extended accordingly, and the Method tab states this on screen.

| Proposal assumed | Dataset actually contains |
|---|---|
| 888 plant-month records, 2025 only | **2,664 records, 2023–2025** |
| No cost data | Chemical, maintenance, energy, total opex, cost per m³ |
| No tariff → no financial estimates | `Tarif_Purata_RM_m3`, RM 0.93–1.37 |
| Real vs apparent losses inseparable | Explicit physical / commercial split per plant |
| Negative NRW records to be flagged | **None** — the anomaly does not occur |
| Median NRW 48.9–55.6% per month | Median plant rate **37.2%**; system rate 32.1% |

Consequences: financial and multi-year analysis is in scope where the proposal
had ruled it out, and the negative-record limitation does not arise. The
proposal's central argument — that rate-based and volume-based prioritisation
produce materially different intervention orders — is **confirmed, and more
strongly than anticipated**: the two measures are negatively correlated.

---

## Limitations

- **Repair cost is not in the dataset.** LIPS ranks by recoverable *volume*
  weighted by condition, not by cost per m³ recovered. Leak concentration is a
  partial proxy, not a substitute for repair costing.
- **The physical/commercial split is a published apportionment**, used as given.
  It is not derived here from a formal IWA water-balance audit.
- **Correlation across plants is not causal.** Plants differ in network length,
  terrain, age and pressure simultaneously.
- **Three years is short.** Trend and year-specific event cannot be fully
  separated; per-plant anomaly detection rests on 36 monthly observations.
- **Financial figures are tariff-based, not full-cost.** Operating cost covers
  energy, chemicals and maintenance only — staff, capital charges and
  depreciation are absent, so operating margins are overstated relative to full
  cost recovery.
- **The model has no ground truth**, so a large residual is a lead, not a
  diagnosis. It is also only as good as the feature set: terrain, soil and
  historic construction quality are not in the data, so the residual bounds
  *where to look* without identifying the cause.
- **Permutation importance is unreliable under collinearity.** Capacity, staff
  count, population and connections all scale with plant size, and the method
  splits credit unpredictably among correlated features. The safe reading is
  that plant size and area type dominate, not that any single column is
  decisive.

### One known cosmetic limitation

Streamlit's slider track paints its fill with an inline `linear-gradient`
carrying the default red accent. Because the colour is baked into an inline
style, it cannot be overridden from CSS without also destroying the fill
proportion indicator, so the slider track stays red in both modes. Every other
accent (chips, tabs, radio, checkbox) is re-pointed at the palette blue, and no
*data* mark is affected — the chart palettes are fully under the design system's
control.

---

## Deploying

The dashboard runs entirely from this folder — `streamlit run app.py` is all it
needs, on any machine with the requirements installed.

If you later want it hosted, [share.streamlit.io](https://share.streamlit.io)
serves it from a repository pointed at `app.py`. Keep the `data/` folder with it:
the host has no build step that would run `refresh.py`, so the app needs the
prepared CSVs present or it will start with nothing to show.
