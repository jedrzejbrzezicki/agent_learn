"""
Forecast Comparison Agent — LangGraph-style (no external dependencies)
=======================================================================
Architecture mirrors LangGraph: TypedDict State + node functions + graph runner.

When LangGraph + HuggingFace API are available, swap:
  - AgentState            → langgraph TypedDict state
  - run_graph()           → StateGraph(...).compile().invoke(...)
  - "LLM verdict" stub    → real HuggingFace InferenceClient call

Tools (agent nodes):
  1. load_data            — reads both CSVs, identifies hierarchies
  2. compute_sum          — total forecast per hierarchy per file
  3. compute_trend        — linear slope of forecast per hierarchy per file
  4. compare_metrics      — flags anomalies (sum diff %, trend reversal)
  5. build_report         — assembles final comparison DataFrame + summary
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from dataclasses import dataclass, field
from typing import Optional
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
HAS_STATSMODELS = True
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FILE_1 = "STRATEGIC_FORECAST_SNOWFLAKE.csv"
FILE_2 = "purina_ea_feb.csv"
OUTPUT  = "outputs\forecast_comparison_report.csv"
HF_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')  # HuggingFace API token (for real LLM verdict)
HIERARCHY = ["AGG_REGION", "REGION", "RANGE_BRAND_NAME"]   # meaningful hier. cols


HIERARCHY = ["AGG_REGION", "REGION", "RANGE_BRAND_NAME", "GA1_NAME"]

PLOTS_DIR  = "outputs"
OUTPUT_XLS = os.path.join("outputs", "forecast_comparison_report.xlsx")

HIERARCHY   = ["AGG_REGION", "REGION", "RANGE_BRAND_NAME", "GA1_NAME"]
ACT_COL     = "ACTUALS"
IGNORE_COLS = {"TIMESTAMP", "UOM", "UNIT_INITIAL"}
SEASON      = 4   # quarterly data

# Flagging thresholds
SUM_DIFF_PCT_THRESHOLD     = 5.0
TREND_DIFF_PCT_THRESHOLD   = 20.0
ACTUALS_DIFF_PCT_THRESHOLD = 8   # above this -> UNREPAIRABLE
REPAIR_IMPROVEMENT_MIN     = 0.01  # model must beat baseline MAPE by >= 10 %

# LLM toggle — LLM is now always enabled


# ─────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────
@dataclass
class AgentState:
    df1: Optional[pd.DataFrame]             = None
    df2: Optional[pd.DataFrame]             = None
    label_v1: str                           = "V1"
    label_v2: str                           = "V2"
    pred_col_v1: str                        = ""
    pred_col_v2: str                        = ""
    hierarchies: list                       = field(default_factory=list)
    sum_metrics: Optional[pd.DataFrame]     = None
    trend_metrics: Optional[pd.DataFrame]   = None
    actuals_metrics: Optional[pd.DataFrame] = None
    comparison_df: Optional[pd.DataFrame]   = None
    repair_results: dict                    = field(default_factory=dict)
    plot_paths: list                        = field(default_factory=list)
    summary: str                            = ""
    errors: list                            = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────────────────────────────

def node_load_data(state: AgentState) -> AgentState:
    """Load both files, normalise to Scenario 0, sum STATUS sub-rows."""
    print("[Node 1/8] Loading & normalizing data...")
    try:
        raw1 = pd.read_csv(FILE_1, parse_dates=["DATE"])
        raw2 = pd.read_csv(FILE_2, parse_dates=["DATE"])

        df1, pc1 = _rename_cols_and_filter(raw1, os.path.basename(FILE_1))
        df2, pc2 = _rename_cols_and_filter(raw2, os.path.basename(FILE_2))
        state.df1, state.pred_col_v1 = df1, pc1
        state.df2, state.pred_col_v2 = df2, pc2
        state.label_v1 = os.path.basename(FILE_1).replace(".csv", "")
        state.label_v2 = os.path.basename(FILE_2).replace(".csv", "")

        hier1 = set(df1.groupby(HIERARCHY).groups.keys())
        hier2 = set(df2.groupby(HIERARCHY).groups.keys())
        state.hierarchies = sorted(hier1 & hier2)

        print(f"  V1 [{state.label_v1}]: {len(df1)} rows, forecast col = '{pc1}'")
        print(f"  V2 [{state.label_v2}]: {len(df2)} rows, forecast col = '{pc2}'")
        print(f"  Shared hierarchies: {len(hier1 & hier2)} / "
              f"V1-only: {len(hier1 - hier2)} / V2-only: {len(hier2 - hier1)}")

        for label, dfi, pc in [(state.label_v1, df1, pc1), (state.label_v2, df2, pc2)]:
            fc = dfi[dfi[pc].notna() & (dfi[pc] > 0)]
            ac = dfi[dfi[ACT_COL].notna() & (dfi[ACT_COL] > 0)]
            fc_range = f"{fc['DATE'].min().date()} -> {fc['DATE'].max().date()}" if len(fc) else "none"
            ac_range = f"{ac['DATE'].min().date()} -> {ac['DATE'].max().date()}" if len(ac) else "none"
            print(f"  [{label}] forecast: {fc_range} | actuals: {ac_range}")

    except Exception as e:
        state.errors.append(f"load_data: {e}")
        import traceback; traceback.print_exc()
    return state


def node_compute_sum(state: AgentState) -> AgentState:
    """Sum of forecast values per hierarchy — positional alignment (no date join)."""
    print("[Node 2/8] Computing forecast sums (positional)...")
    rows = []
    for key in state.hierarchies:
        f1 = _forecast_vals(state.df1, key, state.pred_col_v1)
        f2 = _forecast_vals(state.df2, key, state.pred_col_v2)
        rows.append({
            **_hdict(key),
            "sum_v1":       f1.sum() if len(f1) else np.nan,
            "sum_v2":       f2.sum() if len(f2) else np.nan,
            "n_periods_v1": len(f1),
            "n_periods_v2": len(f2),
        })
    state.sum_metrics = pd.DataFrame(rows)
    return state


def node_compute_trend(state: AgentState) -> AgentState:
    """Linear slope of forecast series per hierarchy."""
    print("[Node 3/8] Computing forecast trends...")
    rows = []
    for key in state.hierarchies:
        rows.append({
            **_hdict(key),
            "slope_v1": _slope(_forecast_vals(state.df1, key, state.pred_col_v1)),
            "slope_v2": _slope(_forecast_vals(state.df2, key, state.pred_col_v2)),
        })
    state.trend_metrics = pd.DataFrame(rows)
    return state


def node_compare_actuals(state: AgentState) -> AgentState:
    """Compare historical actuals — feeds root-cause reasoning."""
    print("[Node 4/8] Comparing actuals...")
    rows = []
    for key in state.hierarchies:
        mask1 = _hmask(state.df1, key)
        mask2 = _hmask(state.df2, key)

        a1 = state.df1.loc[mask1].groupby("DATE")[ACT_COL].sum()
        a2 = state.df2.loc[mask2].groupby("DATE")[ACT_COL].sum()

        both = pd.DataFrame({"a1": a1, "a2": a2}).dropna()
        both = both[(both["a1"] > 0) | (both["a2"] > 0)]

        if both.empty:
            rows.append({**_hdict(key), "actuals_sum_v1": np.nan, "actuals_sum_v2": np.nan,
                         "actuals_diff_pct": np.nan, "actuals_changed": False,
                         "actuals_direction": "none"})
            continue

        s1, s2    = both["a1"].sum(), both["a2"].sum()
        pct       = (s2 - s1) / abs(s1) * 100 if s1 != 0 else np.nan
        changed   = bool(not np.isnan(pct) and abs(pct) > ACTUALS_DIFF_PCT_THRESHOLD)
        direction = ("up" if pct > 0 else "down") if changed else "stable"

        rows.append({**_hdict(key),
                     "actuals_sum_v1":    round(s1, 2),
                     "actuals_sum_v2":    round(s2, 2),
                     "actuals_diff_pct":  round(pct, 3) if not np.isnan(pct) else np.nan,
                     "actuals_changed":   changed,
                     "actuals_direction": direction})
    state.actuals_metrics = pd.DataFrame(rows)
    return state


def node_compare_metrics(state: AgentState) -> AgentState:
    """Flag anomalies + root-cause decision tree."""
    print("[Node 5/8] Comparing metrics & reasoning...")
    df = (state.sum_metrics
          .merge(state.trend_metrics,   on=HIERARCHY)
          .merge(state.actuals_metrics, on=HIERARCHY))

    # ── Flags ────────────────────────────────────────────────────────
    safe_sum              = df["sum_v1"].replace(0, np.nan)
    df["sum_diff_pct"]    = ((df["sum_v2"] - df["sum_v1"]) / safe_sum.abs() * 100).round(3)
    df["flag_sum"]        = df["sum_diff_pct"].abs() > SUM_DIFF_PCT_THRESHOLD

    df["flag_trend_sign"] = (
        np.sign(df["slope_v1"].fillna(0)) != np.sign(df["slope_v2"].fillna(0))
    )
    safe_slope             = df["slope_v1"].replace(0, np.nan)
    df["slope_diff_pct"]   = ((df["slope_v2"] - df["slope_v1"]) / safe_slope.abs() * 100).round(3)
    df["flag_trend_mag"]   = df["slope_diff_pct"].abs() > TREND_DIFF_PCT_THRESHOLD

    df["FLAGGED"] = df["flag_sum"] | df["flag_trend_sign"] | df["flag_trend_mag"]

    # ── Root-cause decision tree ──────────────────────────────────────
    #
    # Decision logic:
    #   actuals changed?
    #     YES -> forecast same direction?  YES -> ACTUALS_DRIVEN
    #                                      NO  -> MIXED_SIGNAL
    #     NO  -> trend sign flipped?       YES -> MODEL_CHANGE
    #            large sum diff (>20%)?    YES -> STRUCTURAL_SHIFT
    #                                      NO  -> MINOR_DRIFT

    def root_cause(row):
        if not row["FLAGGED"]:
            return "OK"

        from huggingface_hub import InferenceClient
        client = InferenceClient(
            model="meta-llama/Meta-Llama-3-8B-Instruct", token=HF_TOKEN
        )
        response = client.chat_completion(
            messages=[{"role": "user", "content": _build_prompt(row)}],
            max_tokens=150
        )
        return response.choices[0].message.content

    df["FLAG_REASON"] = df.apply(lambda r: " | ".join(filter(None, [
        f"sum_diff={r['sum_diff_pct']:+.1f}%" if r["flag_sum"]        else "",
        "trend_sign_flip"                      if r["flag_trend_sign"] else "",
        f"slope_diff={r['slope_diff_pct']:+.1f}%" if r["flag_trend_mag"] else "",
    ])) or "OK", axis=1)

    df["ROOT_CAUSE"] = df.apply(root_cause, axis=1)
    state.comparison_df = df
    return state


def node_triage_and_repair(state: AgentState) -> AgentState:
    """
    Agent decision node — for each flagged hierarchy decides:

    UNREPAIRABLE  actuals differ > ACTUALS_DIFF_PCT_THRESHOLD.
                  Data mismatch — no repair attempted.

    REPAIRABLE    actuals stable, forecast diverged.
                  Fit 3 models on V1 actuals, pick closest to V2.
                  If none improve over V1 baseline -> DEFAULT.

    Adds columns to comparison_df:
      TRIAGE        UNREPAIRABLE / REPAIRABLE / OK
      REPAIR_MODEL  winning model name, DEFAULT, or None
      REPAIR_MAPE   MAPE of winning model vs V2 target
      BASELINE_MAPE MAPE of V1 forecast vs V2 target (before repair)
    """
    print("[Node 6/8] Triaging and repairing flagged hierarchies...")

    flagged = state.comparison_df[state.comparison_df["FLAGGED"]]
    if flagged.empty:
        print("  No flagged hierarchies.")
        _write_repair_cols(state)
        return state

    for _, row in flagged.iterrows():
        key     = tuple(row[c] for c in HIERARCHY)
        key_str = " / ".join(str(k) for k in key)
        act_pct = abs(row.get("actuals_diff_pct", 0) or 0)

        # ── Triage decision ──────────────────────────────────────────
        if row.get("actuals_changed", False) and act_pct > ACTUALS_DIFF_PCT_THRESHOLD:
            state.repair_results[key] = {
                "triage":        "UNREPAIRABLE",
                "reason":        f"actuals differ {act_pct:.1f}% — data mismatch not model issue",
                "best_model":    None,
                "best_mape":     None,
                "baseline_mape": None,
                "model_mapes":   {},
                "repaired_fc":   None,
            }
            print(f"  UNREPAIRABLE  {key_str}  (actuals diff={act_pct:.1f}%)")
            continue

        # ── Gather inputs ────────────────────────────────────────────
        mask1     = _hmask(state.df1, key)
        actuals_s = (state.df1.loc[mask1]
                     .groupby("DATE")[ACT_COL].sum()
                     .sort_index())
        actuals_s = actuals_s[actuals_s > 0]
        y         = actuals_s.values.astype(float)

        target_fc = _forecast_vals(state.df2, key, state.pred_col_v2)
        n_periods = len(target_fc)

        if len(y) < SEASON * 2 or n_periods == 0:
            state.repair_results[key] = {
                "triage": "REPAIRABLE", "reason": "insufficient history",
                "best_model": "DEFAULT", "best_mape": None,
                "baseline_mape": None, "model_mapes": {}, "repaired_fc": None,
            }
            print(f"  DEFAULT       {key_str}  (insufficient history: {len(y)} actuals)")
            continue

        # ── Fit 3 models ─────────────────────────────────────────────
        candidates = {}

        if HAS_STATSMODELS:
            # 1. ARIMA(0,1,1)
            try:
                fit = ARIMA(y, order=(0, 1, 1)).fit()
                candidates["ARIMA(0,1,1)"] = np.maximum(fit.forecast(steps=n_periods), 0)
            except Exception as e:
                print(f"    ARIMA failed: {e}")

            # 2. ETS — Holt-Winters additive trend + seasonal if enough history
            try:
                use_seasonal = len(y) >= SEASON * 2
                fit = ExponentialSmoothing(
                    y,
                    trend="add",
                    damped_trend=True,
                    seasonal="add"        if use_seasonal else None,
                    seasonal_periods=SEASON if use_seasonal else None,
                ).fit(optimized=True)
                candidates["ETS"] = np.maximum(fit.forecast(n_periods), 0)
            except Exception as e:
                print(f"    ETS failed: {e}")

        else:
            # Fallback manual ARIMA(0,1,1) via scipy
            try:
                from scipy.optimize import minimize_scalar
                dy = np.diff(y)
                def ima_sse(theta):
                    eps = np.zeros(len(dy))
                    for t in range(1, len(dy)):
                        eps[t] = dy[t] - theta * eps[t - 1]
                    return np.sum(eps ** 2)
                theta    = minimize_scalar(ima_sse, bounds=(-0.99, 0.99), method="bounded").x
                eps      = np.zeros(len(dy))
                for t in range(1, len(dy)):
                    eps[t] = dy[t] - theta * eps[t - 1]
                last_eps, last_y = eps[-1], y[-1]
                fc = np.zeros(n_periods)
                for h in range(n_periods):
                    fc[h]    = last_y + theta * (last_eps if h == 0 else 0.0)
                    last_y   = fc[h]
                candidates["ARIMA(0,1,1)"] = np.maximum(fc, 0)
            except Exception as e:
                print(f"    ARIMA fallback failed: {e}")

            # Fallback manual ETS via scipy
            try:
                from scipy.optimize import minimize
                n_init = max(4, len(y) // 3)
                slope0, intercept0, *_ = stats.linregress(np.arange(n_init), y[:n_init])

                def ets_mse(params):
                    alpha, beta, phi = params
                    if not (0 < alpha < 1 and 0 < beta < 1 and 0.8 < phi <= 1.0):
                        return 1e12
                    ll, lb = float(intercept0), float(slope0)
                    sse = 0.0
                    for yt in y:
                        err    = yt - (ll + phi * lb)
                        sse   += err ** 2
                        ll, lb = ll + phi * lb + alpha * err, phi * lb + beta * alpha * err
                    return sse / len(y)

                r          = minimize(ets_mse, x0=[0.3, 0.1, 0.98],
                                      bounds=[(0.01, 0.99), (0.01, 0.99), (0.8, 1.0)],
                                      method="L-BFGS-B")
                alpha, beta, phi = r.x
                ll, lb = float(intercept0), float(slope0)
                for yt in y:
                    err    = yt - (ll + phi * lb)
                    ll, lb = ll + phi * lb + alpha * err, phi * lb + beta * alpha * err
                fc = np.array([ll + lb * sum(phi ** i for i in range(1, h + 2))
                               for h in range(n_periods)])
                candidates["ETS"] = np.maximum(fc, 0)
            except Exception as e:
                print(f"    ETS fallback failed: {e}")

        # 3. Seasonal Naive — always available, no library needed
        try:
            last_cycle = y[-SEASON:]
            candidates["SEASONAL_NAIVE"] = np.tile(
                last_cycle, int(np.ceil(n_periods / SEASON))
            )[:n_periods]
        except Exception as e:
            print(f"    Seasonal Naive failed: {e}")

        # ── Baseline MAPE: V1 forecast vs V2 target ──────────────────
        v1_fc         = _forecast_vals(state.df1, key, state.pred_col_v1)
        min_len       = min(len(v1_fc), len(target_fc))
        baseline_mape = _mape(v1_fc[:min_len], target_fc[:min_len]) if min_len > 0 else np.nan

        # ── Model MAPEs ──────────────────────────────────────────────
        model_mapes = {}
        for name, fc in candidates.items():
            ml = min(len(fc), len(target_fc))
            if ml > 0:
                model_mapes[name] = round(_mape(fc[:ml], target_fc[:ml]), 3)

        # ── Pick winner ──────────────────────────────────────────────
        best_model, best_mape, best_fc = "DEFAULT", None, None
        if model_mapes:
            winner = min(model_mapes, key=model_mapes.get)
            w_mape = model_mapes[winner]
            if (not np.isnan(baseline_mape) and
                    w_mape < baseline_mape * (1 - REPAIR_IMPROVEMENT_MIN)):
                best_model = winner
                best_mape  = w_mape
                best_fc    = candidates[winner]

        state.repair_results[key] = {
            "triage":        "REPAIRABLE",
            "reason":        "actuals stable, forecast diverged",
            "best_model":    best_model,
            "best_mape":     best_mape,
            "baseline_mape": round(float(baseline_mape), 3) if not np.isnan(baseline_mape) else None,
            "model_mapes":   model_mapes,
            "repaired_fc":   best_fc,
        }
        print(f"  REPAIRABLE    {key_str}")
        print(f"    baseline={baseline_mape:.1f}%  models={model_mapes}  winner={best_model}")

    _write_repair_cols(state)
    return state


def node_plot_flagged(state: AgentState) -> AgentState:
    """3-panel plot per flagged hierarchy.
       Panel 1 — forecast overlay + repaired forecast if available (positional)
       Panel 2 — actuals overlay (date axis, both files)
       Panel 3 — period-by-period % diff (V2 vs V1)
    """
    print("[Node 7/8] Plotting flagged hierarchies...")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    flagged = state.comparison_df[state.comparison_df["FLAGGED"]]
    if flagged.empty:
        print("  No flagged hierarchies.")
        return state

    for _, row in flagged.iterrows():
        key   = tuple(row[c] for c in HIERARCHY)
        mask1 = _hmask(state.df1, key)
        mask2 = _hmask(state.df2, key)

        def agg(df, mask, col):
            return (df.loc[mask].groupby("DATE")[col].sum()
                    .reset_index().sort_values("DATE"))

        f1_all = agg(state.df1, mask1, state.pred_col_v1)
        f2_all = agg(state.df2, mask2, state.pred_col_v2)
        a1_all = agg(state.df1, mask1, ACT_COL)
        a2_all = agg(state.df2, mask2, ACT_COL)

        f1_fc = f1_all[f1_all[state.pred_col_v1] > 0].reset_index(drop=True)
        f2_fc = f2_all[f2_all[state.pred_col_v2] > 0].reset_index(drop=True)
        a1_ac = a1_all[a1_all[ACT_COL] > 0]
        a2_ac = a2_all[a2_all[ACT_COL] > 0]

        repair      = state.repair_results.get(key, {})
        triage      = repair.get("triage", "")
        repaired_fc = repair.get("repaired_fc")
        best_model  = repair.get("best_model", "DEFAULT")

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        triage_color = "#7f8c8d" if triage == "UNREPAIRABLE" else "#c0392b"
        fig.suptitle(
            " / ".join(str(k) for k in key)
            + f"\n[{triage}]  {row['ROOT_CAUSE']}",
            fontsize=8, fontweight="bold", color=triage_color, wrap=True
        )
        fmt = plt.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if abs(x) >= 1e6 else f"{x/1e3:.0f}K"
        )

        # ── Panel 1: Forecast overlay (positional) ───────────────────
        ax = axes[0]
        ax.plot(np.arange(len(f1_fc)), f1_fc[state.pred_col_v1].values,
                label=state.label_v1, color="#2980b9", lw=1.8)
        ax.plot(np.arange(len(f2_fc)), f2_fc[state.pred_col_v2].values,
                label=state.label_v2, color="#e74c3c", lw=1.8, ls="--")

        if repaired_fc is not None and best_model != "DEFAULT":
            ax.plot(np.arange(len(repaired_fc)), repaired_fc,
                    label=f"Repaired ({best_model})", color="#8e44ad", lw=1.5, ls=":")

        if len(f1_fc):
            step  = max(1, len(f1_fc) // 6)
            ticks = list(range(0, len(f1_fc), step))
            ax.set_xticks(ticks)
            ax.set_xticklabels(
                [f1_fc["DATE"].iloc[i].strftime("%Y-%m") for i in ticks],
                rotation=35, ha="right", fontsize=7
            )
        n_diff = len(f2_fc) - len(f1_fc)
        if n_diff:
            ax.annotate(
                f"V2 has {abs(n_diff)} {'more' if n_diff > 0 else 'fewer'} periods",
                xy=(0.5, 0.02), xycoords="axes fraction",
                ha="center", fontsize=7, color="gray", style="italic"
            )
        ax.set_title("Forecast (positional alignment)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(fmt)

        # ── Panel 2: Actuals overlay ─────────────────────────────────
        ax2 = axes[1]
        if not a1_ac.empty:
            ax2.plot(a1_ac["DATE"], a1_ac[ACT_COL],
                     label=state.label_v1, color="#27ae60", lw=1.5)
        if not a2_ac.empty:
            ax2.plot(a2_ac["DATE"], a2_ac[ACT_COL],
                     label=state.label_v2, color="#f39c12", lw=1.5, ls="--")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax2.get_xticklabels(), rotation=35, ha="right", fontsize=7)
        changed = row.get("actuals_changed", False)
        act_pct = row.get("actuals_diff_pct", 0) or 0
        ax2.set_xlabel(
            f"ACTUALS CHANGED {act_pct:+.1f}%" if changed else "actuals stable",
            fontsize=7, color="#c0392b" if changed else "#27ae60"
        )
        ax2.set_title("Actuals Comparison", fontsize=8)
        if not a1_ac.empty or not a2_ac.empty:
            ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(fmt)

        # ── Panel 3: Period % diff ───────────────────────────────────
        ax3 = axes[2]
        min_len = min(len(f1_fc), len(f2_fc))
        if min_len > 0:
            v1 = f1_fc[state.pred_col_v1].values[:min_len]
            v2 = f2_fc[state.pred_col_v2].values[:min_len]
            with np.errstate(invalid="ignore", divide="ignore"):
                pct = np.where(v1 != 0, (v2 - v1) / np.abs(v1) * 100, np.nan)
            colors = ["#e74c3c" if (x or 0) > 0 else "#2980b9" for x in np.nan_to_num(pct)]
            ax3.bar(np.arange(min_len), pct, color=colors, alpha=0.75, width=0.8)
            ax3.axhline(0, color="black", lw=0.8)
            ax3.axhline( SUM_DIFF_PCT_THRESHOLD, color="orange", lw=1, ls=":",
                        label=f"±{SUM_DIFF_PCT_THRESHOLD}%")
            ax3.axhline(-SUM_DIFF_PCT_THRESHOLD, color="orange", lw=1, ls=":")
            ax3.legend(fontsize=7)
        ax3.set_title("Forecast % diff per period (V2 vs V1)", fontsize=8)
        ax3.set_xlabel("Period index", fontsize=7)
        ax3.set_ylabel("% diff", fontsize=7)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = "_".join(str(k) for k in key).replace(" ", "_").replace("/", "-").replace(",", "")[:120]
        path  = os.path.join(PLOTS_DIR, f"{fname}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        state.plot_paths.append(path)
        print(f"  Saved: {os.path.basename(path)}")

    return state


def node_build_report(state: AgentState) -> AgentState:
    print("[Node 8/8] Building report...")
    df      = state.comparison_df
    total   = len(df)
    flagged = int(df["FLAGGED"].sum())
    ok      = total - flagged

    unrepairable = int((df["TRIAGE"] == "UNREPAIRABLE").sum())
    repairable   = int((df["TRIAGE"] == "REPAIRABLE").sum())
    repaired     = int(df[df["TRIAGE"] == "REPAIRABLE"]["REPAIR_MODEL"]
                       .apply(lambda x: x not in (None, "DEFAULT") if pd.notna(x) else False).sum())
    default      = repairable - repaired

    root_counts = {k: int(df["ROOT_CAUSE"].str.contains(k, na=False).sum())
                   for k in ["ACTUALS_DRIVEN", "MODEL_CHANGE", "STRUCTURAL_SHIFT",
                              "MIXED_SIGNAL", "MINOR_DRIFT"]}

    # ── LLM executive summary ─────────────────────────────────────────
    from huggingface_hub import InferenceClient
    client  = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=HF_TOKEN)
    prompt  = (f"Forecasting analyst report: {flagged}/{total} hierarchies flagged "
               f"comparing {state.label_v1} vs {state.label_v2}. "
               f"Root causes: {root_counts}. "
               f"Triage: {unrepairable} unrepairable, {repaired} repaired, {default} defaulted. "
               f"Write a concise 3-sentence executive summary.")
    verdict = client.chat_completion(
        messages=[{"role": "user", "content": prompt}], max_tokens=300
    ).choices[0].message.content

    state.summary = (
        f"\n{'='*65}\n"
        f"  FORECAST COMPARISON + REPAIR REPORT\n"
        f"{'='*65}\n"
        f"  V1 : {state.label_v1}\n"
        f"  V2 : {state.label_v2}\n"
        f"  Hierarchies analysed : {total}\n"
        f"  OK                   : {ok}\n"
        f"  FLAGGED              : {flagged}\n"
        f"  -- UNREPAIRABLE      : {unrepairable}\n"
        f"  -- REPAIRABLE        : {repairable}\n"
        f"     -- model fixed    : {repaired}\n"
        f"     -- DEFAULT        : {default}\n"
        + "".join(f"  -- {k.replace('_', ' ').title():<22}: {v}\n"
                  for k, v in root_counts.items())
        + f"  Plots saved          : {len(state.plot_paths)} -> {PLOTS_DIR}\n"
        f"\n  Agent verdict:\n  {verdict}\n"
        f"{'='*65}\n"
    )

    # ── Round numeric cols ────────────────────────────────────────────
    for col in ["sum_v1", "sum_v2", "slope_v1", "slope_v2",
                "actuals_sum_v1", "actuals_sum_v2"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    # ── Two-sheet Excel ───────────────────────────────────────────────
    # Sheet 1: all hierarchies
    # Sheet 2: flagged only, UNREPAIRABLE rows first
    os.makedirs(PLOTS_DIR, exist_ok=True)
    flagged_df = df[df["FLAGGED"]].copy()
    flagged_df = flagged_df.sort_values(
        ["TRIAGE", "ROOT_CAUSE"], ascending=[False, True]
    )

    with pd.ExcelWriter(OUTPUT_XLS, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Hierarchies", index=False)
        flagged_df.to_excel(writer, sheet_name="Flagged", index=False)

    print(f"  Report saved -> {OUTPUT_XLS}")
    return state


# ─────────────────────────────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────────────────────────────

GRAPH = [
    node_load_data,
    node_compute_sum,
    node_compute_trend,
    node_compare_actuals,
    node_compare_metrics,
    node_triage_and_repair,
    node_plot_flagged,
    node_build_report,
]

def run_graph() -> AgentState:
    state = AgentState()
    for node in GRAPH:
        state = node(state)
        if state.errors:
            print(f"  !! Errors: {state.errors}")
    return state


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _rename_cols_and_filter(df: pd.DataFrame, filename: str) -> tuple[pd.DataFrame, str]:
    """
    Normalise a raw file to a clean, comparable DataFrame:
      1. Drop irrelevant columns (TIMESTAMP, UOM, UNIT_INITIAL)
      2. Filter to Scenario 0 (or keep all if no SCENARIO column)
      3. Filter MANUAL_INPUT == 0 if column exists, then drop it
      4. Detect forecast column (PREDICTION or FORECAST_UNIVARIATE)
      5. Limit to first 60 forecast months from horizon
      6. Convert dates to quarterly periods
      7. Sum STATUS sub-rows -> one row per HIERARCHY+DATE
    """
    df = df.drop(columns=[c for c in IGNORE_COLS if c in df.columns])

    if "SCENARIO" in df.columns:
        df = df[df["SCENARIO"] == "Scenario 0"].copy()
        df = df.drop(columns=["SCENARIO"])

    if "MANUAL_INPUT" in df.columns:
        df = df[df["MANUAL_INPUT"].isin([0, "0"])].copy()
        df = df.drop(columns=["MANUAL_INPUT"])

    if "PREDICTION" in df.columns:
        pred_col = "PREDICTION"
    elif "FORECAST_UNIVARIATE" in df.columns:
        pred_col = "FORECAST_UNIVARIATE"
    else:
        raise ValueError(f"No forecast column found in {filename}. "
                         f"Expected 'PREDICTION' or 'FORECAST_UNIVARIATE'.")

    # Limit to first 60 forecast months from horizon
    horizon_date = df[df[pred_col] > 0]["DATE"].min()
    if pd.notna(horizon_date):
        cutoff_date = horizon_date + pd.DateOffset(months=60)
        df = df[df["DATE"] <= cutoff_date].copy()

    # Convert to quarterly periods
    df["DATE"] = df["DATE"].dt.to_period("Q").dt.to_timestamp()

    # Sum STATUS sub-rows -> true hierarchy grain
    keep_cols = [c for c in HIERARCHY + ["DATE", pred_col, ACT_COL] if c in df.columns]
    df = (df[keep_cols]
          .groupby(HIERARCHY + ["DATE"], as_index=False)
          .agg({pred_col: "sum", ACT_COL: "sum"}))

    print(f"  [{filename}] normalized: {len(df)} rows, "
          f"pred_col='{pred_col}', "
          f"hierarchies={df.groupby(HIERARCHY).ngroups}")
    return df, pred_col


def _write_repair_cols(state: AgentState) -> None:
    """Write TRIAGE / REPAIR_MODEL / REPAIR_MAPE / BASELINE_MAPE into comparison_df."""
    def _get(row, field, default=None):
        r = state.repair_results.get(tuple(row[c] for c in HIERARCHY))
        return r[field] if r else default

    state.comparison_df["TRIAGE"] = state.comparison_df.apply(
        lambda r: _get(r, "triage", "OK" if not r["FLAGGED"] else "FLAGGED"), axis=1)
    state.comparison_df["REPAIR_MODEL"] = state.comparison_df.apply(
        lambda r: _get(r, "best_model"), axis=1)
    state.comparison_df["REPAIR_MAPE"] = state.comparison_df.apply(
        lambda r: _get(r, "best_mape"), axis=1)
    state.comparison_df["BASELINE_MAPE"] = state.comparison_df.apply(
        lambda r: _get(r, "baseline_mape"), axis=1)


def _hmask(df: pd.DataFrame, key: tuple) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in zip(HIERARCHY, key):
        mask &= (df[col] == val)
    return mask

def _hdict(key: tuple) -> dict:
    return dict(zip(HIERARCHY, key))

def _forecast_vals(df: pd.DataFrame, key: tuple, pred_col: str) -> np.ndarray:
    """Sorted forecast values for a hierarchy, excluding zeros/nulls."""
    mask = _hmask(df, key)
    sub  = df.loc[mask].sort_values("DATE")
    vals = sub[pred_col].dropna()
    return vals[vals > 0].values

def _slope(vals: np.ndarray) -> float:
    if len(vals) < 2:
        return np.nan
    s, *_ = stats.linregress(np.arange(len(vals)), vals)
    return round(float(s), 6)

def _mape(pred: np.ndarray, actual: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nanmean(
            np.where(actual != 0,
                     np.abs(pred - actual) / np.abs(actual),
                     np.nan)
        ) * 100)

def _build_prompt(row: pd.Series) -> str:
    """Structured prompt for LLM root-cause call"""
    hier = " / ".join(str(row[c]) for c in HIERARCHY if c in row.index)
    return (
        f"You are a forecasting analyst reviewing a flagged hierarchy.\n\n"
        f"Hierarchy: {hier}\n"
        f"Forecast sum V1: {row.get('sum_v1', 'N/A')}\n"
        f"Forecast sum V2: {row.get('sum_v2', 'N/A')}\n"
        f"Forecast sum diff: {row.get('sum_diff_pct', 'N/A'):+.1f}%\n"
        f"Trend slope V1: {row.get('slope_v1', 'N/A')}\n"
        f"Trend slope V2: {row.get('slope_v2', 'N/A')}\n"
        f"Trend sign flipped: {row.get('flag_trend_sign', False)}\n"
        f"Actuals sum V1: {row.get('actuals_sum_v1', 'N/A')}\n"
        f"Actuals sum V2: {row.get('actuals_sum_v2', 'N/A')}\n"
        f"Actuals diff: {row.get('actuals_diff_pct', 'N/A')}\n"
        f"Actuals changed significantly: {row.get('actuals_changed', False)}\n\n"
        f"In one sentence, what is the most likely root cause of this forecast change?"
    )


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    state = run_graph()
    print(state.summary)
    if state.comparison_df is not None:
        flagged = state.comparison_df[state.comparison_df["FLAGGED"]]
        if not flagged.empty:
            cols = HIERARCHY + ["sum_diff_pct", "actuals_diff_pct", "actuals_changed",
                                "TRIAGE", "REPAIR_MODEL", "REPAIR_MAPE",
                                "BASELINE_MAPE", "ROOT_CAUSE"]
            print(flagged[cols].to_string(index=False))