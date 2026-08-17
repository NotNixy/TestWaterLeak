"""Independent verification.

Recomputes every headline figure straight from the RAW CSV, using a separate
parsing path from prepare_data.py, and asserts agreement with the prepared
artefacts the dashboard reads. A bug shared by both paths would have to be
introduced twice to survive this.
"""
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Read the same raw inputs the pipeline reads, and take the focus year from the
# data. Nothing here may be pinned to a particular year: a refresh that adds a
# new year must be verifiable without editing this file.
from pathlib import Path as _P
RAW_DIR = _P(__file__).parent / "data" / "raw"
RAW_FILES = sorted(RAW_DIR.glob("*.csv"))
if not RAW_FILES:
    raise SystemExit("No raw CSV found in data/raw/ — run prepare_data.py first.")
fails, checks = [], 0


def ok(name, a, b, tol=1e-6, rel=False):
    global checks
    checks += 1
    d = abs(a - b)
    good = (d / max(abs(b), 1e-9) < tol) if rel else (d <= tol)
    print(f"  {'PASS' if good else 'FAIL'}  {name:52s} {a:>18,.4f} vs {b:>18,.4f}")
    if not good:
        fails.append(name)


# --- Independent parse: regex-strip, no shared helper --------------------
# Deliberately does NOT import dataloader: a bug shared by both paths would have
# to be written twice to survive. Duplicate plant-months are dropped the same
# way the loader does (last wins) so row counts are comparable.
raw = pd.concat([pd.read_csv(f, dtype=str) for f in RAW_FILES], ignore_index=True)
raw = raw.drop_duplicates(["Nama_Loji", "Tarikh"], keep="last").reset_index(drop=True)


def num(col):
    return raw[col].map(lambda v: float(re.sub(r"[,%]", "", str(v))))


YEAR = int(num("Tahun").max())
print(f"Verifying focus year {YEAR} from {len(RAW_FILES)} raw file(s)")
year_mask = num("Tahun") == YEAR
prod = num("Pengeluaran_m3")[year_mask]
billed = num("Jumlah_Dibilkan_m3")[year_mask]
nrw = num("NRW_m3")[year_mask]
phys = num("Kehilangan_Fizikal_m3")[year_mask]
tariff = num("Tarif_Purata_RM_m3")[year_mask]
plants = raw["Nama_Loji"][year_mask]

prep_m = pd.read_csv("data/nrw_plant_month.csv")
prep_y = pd.read_csv("data/nrw_plant_year.csv")
py = prep_y[prep_y.year == YEAR]

print(f"\n=== Totals, {YEAR} ===")
ok("total production m3", prod.sum(), py.production_m3.sum(), 1)
ok("total billed m3", billed.sum(), py.billed_m3.sum(), 1)
ok("total NRW m3", nrw.sum(), py.nrw_m3.sum(), 1)
ok("total physical loss m3", phys.sum(), py.physical_loss_m3.sum(), 1)
ok("system NRW pct", nrw.sum() / prod.sum() * 100, py.nrw_m3.sum() / py.production_m3.sum() * 100, 1e-9)
ok("NRW value RM", (nrw * tariff).sum(), py.nrw_value_rm.sum(), 1.0)
ok("physical share pct", phys.sum() / nrw.sum() * 100,
   py.physical_loss_m3.sum() / py.nrw_m3.sum() * 100, 1e-9)

print("\n=== Structure ===")
ok("plant count", plants.nunique(), len(py), 0)
ok("plant-month rows in year", int(year_mask.sum()),
   plants.nunique() * num("Bulan_No")[year_mask].nunique(), 0)
ok("all-year rows", len(raw), len(prep_m), 0)

# --- Per-plant aggregation from scratch ----------------------------------
ind = (pd.DataFrame({"plant": plants, "prodv": prod, "nrw": nrw, "phys": phys})
       .groupby("plant", as_index=False).sum())
ind["pct"] = ind.nrw / ind.prodv * 100
merged = ind.merge(py[["plant", "nrw_m3", "nrw_pct", "physical_loss_m3"]], on="plant")

print("\n=== Per-plant agreement (74 plants) ===")
ok("max abs diff, plant NRW m3", (merged.nrw - merged.nrw_m3).abs().max(), 0.0, 1)
ok("max abs diff, plant NRW pct", (merged.pct - merged.nrw_pct).abs().max(), 0.0, 1e-6)
ok("max abs diff, plant physical m3",
   (merged.phys - merged.physical_loss_m3).abs().max(), 0.0, 1)

# --- The headline claim on the Rate vs Volume tab ------------------------
print("\n=== Rate-vs-volume claim ===")
ind["rate_rank"] = ind.pct.rank(ascending=False, method="min")
ind["vol_rank"] = ind.nrw.rank(ascending=False, method="min")
t_rate = ind.nsmallest(10, "rate_rank")
t_vol = ind.nsmallest(10, "vol_rank")
rho_ind = spearmanr(ind.pct, ind.nrw).statistic
rho_prep = spearmanr(py.nrw_pct, py.nrw_m3).statistic
ok("spearman rho (rate vs volume)", rho_ind, rho_prep, 1e-9)
ok("top-10 overlap between queues", len(set(t_rate.plant) & set(t_vol.plant)), 0, 0)
# Compared against the prepared artefact, not a frozen number: the true value
# legitimately changes when a new year arrives, and a literal here would fail
# every refresh for the wrong reason.
py_rate = py.nsmallest(10, "rate_rank")
py_vol = py.nsmallest(10, "volume_rank")
ok("water in volume queue / rate queue (x)",
   t_vol.nrw.sum() / t_rate.nrw.sum(),
   py_vol.nrw_m3.sum() / py_rate.nrw_m3.sum(), 0.02)
ok("top-10 share of all NRW (pct)",
   ind.nlargest(10, "nrw").nrw.sum() / ind.nrw.sum() * 100,
   py.nlargest(10, "nrw_m3").nrw_m3.sum() / py.nrw_m3.sum() * 100, 0.01)
checks += 1
_ratio = t_vol.nrw.sum() / t_rate.nrw.sum()
print(f"  {'PASS' if _ratio > 1 else 'FAIL'}  "
      f"{'volume queue holds more water than rate queue':52s} "
      f"{_ratio:>18,.2f}x")
if _ratio <= 1:
    fails.append("volume queue no longer dominates")
print(f"    rho = {rho_ind:.4f}  (negative => rate and volume rank plants oppositely)")

# --- LIPS behaviour -------------------------------------------------------
print("\n=== LIPS ===")
COMPS = ["physical_loss_m3", "nrw_per_km_m3", "nrw_pct", "bursts_per_100km",
         "asset_age_index", "physical_share_pct"]
W = {"physical_loss_m3": 35, "nrw_per_km_m3": 15, "nrw_pct": 13,
     "bursts_per_100km": 12, "asset_age_index": 15, "physical_share_pct": 10}


def lips_of(df, w):
    tot = sum(w.values())
    s = pd.Series(0.0, index=df.index)
    for c, wt in w.items():
        s += df[c].rank(pct=True, method="average") * 100 * (wt / tot)
    return s


recomputed = lips_of(py, W)
ok("max abs diff, LIPS score", (recomputed - py.lips).abs().max(), 0.0, 0.01)
ok("LIPS bounded 0-100", float(((py.lips < 0) | (py.lips > 100)).sum()), 0.0, 0)
ok("LIPS ranks are 1..n unique", py.lips_rank.nunique(), len(py), 0)

# Weight sensitivity: an all-volume weighting must reproduce the volume order.
vol_only = lips_of(py, {"physical_loss_m3": 100})
agree = spearmanr(vol_only, py.physical_loss_m3).statistic
ok("volume-only weighting reproduces volume rank", agree, 1.0, 1e-9)

# The sliders must actually move the ranking, or the control is decorative.
rate_heavy = lips_of(py, {"nrw_pct": 100})
moved = int((rate_heavy.rank(ascending=False, method="min")
             != py.lips_rank.values).sum())
checks += 1
print(f"  {'PASS' if moved > 40 else 'FAIL'}  "
      f"{'reweighting changes plant ranks':52s} {moved:>18} of {len(py)} plants move")
if moved <= 40:
    fails.append("reweighting has too little effect")

# --- Financial identities -------------------------------------------------
print("\n=== Financials ===")
ok("NRW value == volume x tariff (per plant, max diff)",
   (py.nrw_value_rm - py.nrw_m3 * py.tariff_rm_m3).abs().max(), 0.0, 2500,
   )  # tariff is a 12-month mean, so a small residual is expected
sunk = py.nrw_sunk_cost_rm.sum()
ok("sunk cost < forgone revenue", float(sunk < py.nrw_value_rm.sum()), 1.0, 0)
# Recomputed from the monthly artefact rather than pinned to a literal.
_m = pd.read_csv("data/nrw_plant_month.csv")
_m = _m[_m.year == YEAR]
ok("sunk cost share of opex (pct)", sunk / py.opex_rm.sum() * 100,
   _m.nrw_sunk_cost_rm.sum() / _m.opex_rm.sum() * 100, 0.01)




# ==========================================================================
# Model checks — added with the Early Warning tab
# ==========================================================================
import json as _json
from pathlib import Path as _Path

if _Path("data/model_metrics.json").exists():
    import train_models as _tm
    from sklearn.model_selection import GroupKFold as _GKF

    met = _json.loads(_Path("data/model_metrics.json").read_text())
    mlp = pd.read_csv("data/ml_plant.csv")
    mlm = pd.read_csv("data/ml_monthly.csv", parse_dates=["date"])

    print("\n=== Model: target leakage ===")
    feats = set(_tm.NUMERIC_FEATURES) | set(_tm.CATEGORICAL_FEATURES)
    ok("no banned feature in the model", float(len(feats & _tm.BANNED)), 0.0, 0)

    # The two trap columns must be provably derived from BILLED volume, which is
    # production minus the target. Re-derive them here rather than trusting the
    # exclusion list.
    d = pd.read_csv("data/nrw_plant_month.csv", parse_dates=["date"])
    days = d.date.dt.days_in_month
    ok("consumption_per_capita IS billed-derived (would leak)",
       (d.consumption_per_capita_l_day - d.billed_m3 * 1000 / d.population_served / days).abs().max(),
       0.0, 0.06)
    ok("revenue_per_account IS billed-derived (would leak)",
       (d.revenue_per_account_rm - d.billed_revenue_rm / d.customer_accounts).abs().max(),
       0.0, 0.01)
    for c in ("consumption_per_capita_l_day", "revenue_per_account_rm"):
        checks += 1
        good = c not in feats
        print(f"  {'PASS' if good else 'FAIL'}  {('excluded: ' + c):52s} "
              f"{'absent from feature set':>18}")
        if not good:
            fails.append(f"{c} not excluded")

    # Features that divide by PRODUCTION are safe; assert they really do.
    ok("capacity_utilisation is production-derived (safe)",
       (d.capacity_utilisation_pct - d.production_m3 / (d.capacity_m3_day * days) * 100).abs().max(),
       0.0, 0.06)
    ok("energy_intensity is production-derived (safe)",
       (d.energy_intensity_kwh_m3 - d.energy_kwh / d.production_m3).abs().max(), 0.0, 0.001)

    print("\n=== Model: validation integrity ===")
    cv = met["cv_grouped_by_plant"]
    best = met["selected_model"]
    ok("selected model beats mean baseline", float(cv[best]["r2"] > cv["mean"]["r2"]), 1.0, 0)
    ok("selected model has best grouped-CV R2",
       float(cv[best]["r2"] >= max(cv[k]["r2"] for k in ("gb", "ridge"))), 1.0, 0)
    ok("mean baseline R2 is ~0 or negative", float(cv["mean"]["r2"] < 0.01), 1.0, 0)
    ok("selected MAE beats baseline MAE", float(cv[best]["mae"] < cv["mean"]["mae"]), 1.0, 0)

    # GroupKFold must place every plant entirely on one side of each split;
    # otherwise "unseen plants" is a false claim.
    m_eng = _tm.engineer(d)
    X_ = m_eng[_tm.NUMERIC_FEATURES + _tm.CATEGORICAL_FEATURES]
    g_ = m_eng.plant.values
    bleed = 0
    for tr_i, te_i in _GKF(n_splits=5).split(X_, m_eng.nrw_pct.values, g_):
        bleed += len(set(g_[tr_i]) & set(g_[te_i]))
    ok("no plant appears in both train and test", float(bleed), 0.0, 0)

    print("\n=== Model: outputs ===")
    ok("every plant scored", float(mlp.plant.nunique()),
       float(pd.read_csv("data/nrw_plant_year.csv")
             .query("year == @met['focus_year']").plant.nunique()), 0)
    ok("criticality bounded 0-100",
       float(((mlp.criticality < 0) | (mlp.criticality > 100)).sum()), 0.0, 0)
    ok("criticality ranks are 1..n unique", float(mlp.criticality_rank.nunique()),
       float(len(mlp)), 0)
    ok("residual == actual - expected",
       (mlp.unexplained_pp - (mlp.actual_nrw_pct - mlp.expected_nrw_pct)).abs().max(),
       0.0, 1e-6)
    ok("unexplained m3 == residual x production",
       (mlp.unexplained_m3 - mlp.unexplained_pp / 100 * mlp.production_m3).abs().max(),
       0.0, 1.0)
    # Residuals from an out-of-fold fit should be roughly centred, not biased.
    ok("mean residual is near zero (unbiased)", float(mlp.unexplained_pp.mean()), 0.0, 2.0)
    ok("monthly predictions cover every record", float(len(mlm)),
       float(len(pd.read_csv("data/nrw_plant_month.csv"))), 0)
    ok("no NaN predictions", float(mlm.predicted_nrw_pct.isna().sum()), 0.0, 0)

    print("\n=== Model: criticality is not a restatement of LIPS ===")
    yr = pd.read_csv("data/nrw_plant_year.csv")
    _fy = int(met["focus_year"])
    j = mlp[["plant", "criticality"]].merge(
        yr[yr.year == _fy][["plant", "lips", "nrw_m3", "nrw_pct"]], on="plant")
    r_lips = spearmanr(j.criticality, j.lips).statistic
    r_vol = spearmanr(j.criticality, j.nrw_m3).statistic
    checks += 1
    good = abs(r_lips) < 0.8 and abs(r_vol) < 0.8
    print(f"  {'PASS' if good else 'FAIL'}  "
          f"{'criticality is distinct from LIPS/volume':52s} "
          f"rho_lips={r_lips:6.3f}  rho_vol={r_vol:6.3f}")
    if not good:
        fails.append("criticality duplicates LIPS")

    print("\n=== Model: the null deterioration result ===")
    ds = met["deterioration_summary"]
    ok("worsening-trend count matches artefact",
       float(((mlp.trend_p < 0.10) & (mlp.trend_pp_yr > 0)).sum()),
       float(ds["plants_with_significant_worsening"]), 0)
    ok("step-increase count matches artefact",
       float(((mlp.step_p < 0.05) & (mlp.step_shift_pp > 0)).sum()),
       float(ds["plants_with_significant_step_increase"]), 0)
    ok("estate median trend is improving", float(ds["median_trend_pp_yr"] < 0), 1.0, 0)
    print(f"    {ds['plants_with_significant_worsening']} plants worsening, "
          f"{ds['total_anomaly_months']} anomaly months in 2,664 records — "
          f"reported as a null result, not concealed")

print(f"\n{'='*88}")
if fails:
    print(f"{len(fails)} of {checks} CHECKS FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print(f"All {checks} checks passed.")
