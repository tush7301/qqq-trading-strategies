"""Slot 2 research harness: walk-forward model selection inside TRAIN, then validation once.

    python notebooks/02_neural_net/nn_research.py            # families A, C, D (NumPy / sklearn)
    python notebooks/02_neural_net/nn_research.py --lstm     # + families B, C-LSTM (needs torch)

Implements notebooks/02_neural_net/PROTOCOL.md. Only data up to 2021-12-31 is ever
loaded; the test period does not exist for this script. Results are written to
results/strategies/neural_net/research/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qqq import config  # noqa: E402
from qqq import features as F  # noqa: E402
from qqq.backtest import run_backtest  # noqa: E402
from qqq.data import load_prices  # noqa: E402
from qqq.metrics import bootstrap_sharpe_ci, compute_metrics  # noqa: E402
from qqq.strategies.base import BuyAndHold, Strategy  # noqa: E402
from qqq.strategies.neural_net import NeuralNetStrategy  # noqa: E402

OUT = config.RESULTS / "strategies" / "neural_net" / "research"
DEV_END = config.VALIDATION_END                       # nothing after this is ever loaded
FOLDS = [("2010-12-31", "2011-01-01", "2012-12-31"),
         ("2012-12-31", "2013-01-01", "2014-12-31"),
         ("2014-12-31", "2015-01-01", "2016-12-31"),
         ("2016-12-31", "2017-01-01", "2018-12-31")]
VALID = (config.VALIDATION_START, config.VALIDATION_END)


# ------------------------------------------------------------ evaluation core ---
def walk_forward(make, prices) -> dict:
    """Fit a fresh strategy on each fold's fit window; trade its evaluation window.
    Returns joined out-of-fold metrics (2011-2018) and the joined daily series."""
    rets, rfs, pos_all = [], [], []
    for fit_end, ev_start, ev_end in FOLDS:
        strat = make().fit(prices.loc[:fit_end])
        upto = prices.loc[:ev_end]                      # never more data than needed
        pos = strat.generate_positions(upto)
        res = run_backtest(upto, pos).slice(ev_start, ev_end)
        rets.append(res.returns)
        rfs.append(res.rf)
        pos_all.append(res.positions)
    r, rf, p = pd.concat(rets), pd.concat(rfs), pd.concat(pos_all)
    m = compute_metrics(r, rf_daily=rf)
    changes = int((p.diff().abs() > 1e-12).sum())
    return {"sharpe": m["sharpe_ratio"], "cagr": m["annualized_return"],
            "vol": m["annualized_volatility"], "max_dd": m["max_drawdown"],
            "exposure": float(p.mean()), "changes": changes, "_r": r, "_rf": rf}


def validation(make, prices) -> dict:
    strat = make().fit(prices.loc[:config.TRAIN_END])
    pos = strat.generate_positions(prices)              # prices end at DEV_END
    res = run_backtest(prices, pos)
    out = {}
    for name, (a, b) in (("train", (None, config.TRAIN_END)), ("validation", VALID)):
        m = res.slice(a, b).metrics()
        out[name] = m
    sub = res.slice(*VALID)
    out["validation_ci"] = bootstrap_sharpe_ci(sub.returns, n_boot=1000, rf_daily=sub.rf)
    out["_strategy"] = strat
    out["_positions"] = pos
    return out


def auc(y_true, score) -> float:
    from sklearn.metrics import roc_auc_score
    ok = ~(np.isnan(score) | np.isnan(y_true))
    return float(roc_auc_score(y_true[ok], score[ok]))


# --------------------------------------------------------- family D: vol nets ---
def vol_features(prices: pd.DataFrame) -> pd.DataFrame:
    """HAR-style inputs (Corsi 2009) from daily OHLC; all causal, in logs."""
    adj = prices["AdjClose"] / prices["Close"]
    h, lo = prices["High"] * adj, prices["Low"] * adj
    o, c = prices["Open"] * adj, prices["AdjClose"]
    r = c.pct_change()
    pk = (np.log(h / lo) ** 2) / (4 * np.log(2))       # Parkinson daily variance
    sq = r ** 2
    eps = 1e-10
    out = {
        "log_pk_1": np.log(pk + eps),
        "log_pk_5": np.log(pk.rolling(5).mean() + eps),
        "log_pk_22": np.log(pk.rolling(22).mean() + eps),
        "log_rv_5": np.log(sq.rolling(5).mean() + eps),
        "log_rv_22": np.log(sq.rolling(22).mean() + eps),
        "log_rv_66": np.log(sq.rolling(66).mean() + eps),
        "ret_1": r,
        "neg_ret_5": r.clip(upper=0).rolling(5).sum(),  # leverage effect
        "abs_gap": (o / c.shift(1) - 1).abs(),
    }
    return pd.DataFrame(out, index=prices.index)


def future_log_vol(prices: pd.DataFrame, horizon: int) -> pd.Series:
    """TARGET ONLY: log of annualised realised vol over days t+1..t+h, aligned at t."""
    r = prices["AdjClose"].pct_change()
    fwd_var = (r ** 2).rolling(horizon).mean().shift(-horizon)
    return 0.5 * np.log(fwd_var * config.TRADING_DAYS_PER_YEAR + 1e-12)


class VolForecastSizing(Strategy):
    """Always long, sized by min(1, target_vol / forecast vol). The forecast comes from
    `model` in {"rv20" (no model), "ewma", "har" (linear), "mlp"}. target_vol is QQQ's
    realised vol over the fit window (Slot 1's 'train' rule). Rebalance band 10%."""

    name = "nn_vol_sizing"

    def __init__(self, model="mlp", horizon=5, hidden=(16,), n_seeds=5, band=0.10,
                 seed=config.RANDOM_SEED):
        self.model, self.horizon, self.hidden = model, horizon, tuple(hidden)
        self.n_seeds, self.band, self.seed = n_seeds, band, seed

    def fit(self, train_prices):
        r = train_prices["AdjClose"].pct_change()
        self.target_vol = float(r.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR))
        if self.model in ("rv20", "ewma"):
            return self
        X = vol_features(train_prices)
        y = future_log_vol(train_prices, self.horizon)   # built from TRAIN only
        ok = X.notna().all(axis=1) & y.notna()
        self.mu, self.sd = X[ok].mean(), X[ok].std(ddof=0).replace(0, 1)
        Z = ((X[ok] - self.mu) / self.sd).clip(-5, 5).to_numpy()
        t = y[ok].to_numpy()
        if self.model == "har":
            from sklearn.linear_model import Ridge
            self.nets = [Ridge(alpha=1.0).fit(Z, t)]
        else:
            from sklearn.neural_network import MLPRegressor
            n_hold = int(len(Z) * 0.15)                  # time-ordered early stopping
            self.nets = []
            for k in range(self.n_seeds):
                best, best_net, wait = np.inf, None, 0
                net = MLPRegressor(hidden_layer_sizes=self.hidden, alpha=1e-3,
                                   learning_rate_init=1e-3, random_state=self.seed + k)
                for _ in range(300):
                    net.partial_fit(Z[:-n_hold], t[:-n_hold])
                    loss = np.mean((net.predict(Z[-n_hold:]) - t[-n_hold:]) ** 2)
                    if loss < best - 1e-6:
                        import copy
                        best, best_net, wait = loss, copy.deepcopy(net), 0
                    else:
                        wait += 1
                        if wait >= 20:
                            break
                self.nets.append(best_net)
        return self

    def forecast(self, prices) -> pd.Series:
        r = prices["AdjClose"].pct_change()
        ann = np.sqrt(config.TRADING_DAYS_PER_YEAR)
        if self.model == "rv20":
            return r.rolling(20, min_periods=20).std() * ann
        if self.model == "ewma":                         # RiskMetrics lambda = 0.94
            return np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False, min_periods=20).mean()) * ann
        X = vol_features(prices)
        ok = X.notna().all(axis=1)
        out = pd.Series(np.nan, index=prices.index)
        Z = ((X[ok] - self.mu) / self.sd).clip(-5, 5).to_numpy()
        out[ok] = np.exp(np.mean([n.predict(Z) for n in self.nets], axis=0))
        return out

    def generate_positions(self, prices):
        f = self.forecast(prices)
        raw = (self.target_vol / f).clip(upper=1.0)
        vals, cur = [], 0.0
        for v in raw.to_numpy():
            if np.isnan(v):
                vals.append(0.0)
                continue
            if abs(v - cur) > self.band or cur == 0.0:
                cur = v
            vals.append(cur)
        return pd.Series(vals, index=prices.index, name="position")


def vol_forecast_quality(make, prices) -> dict:
    """Out-of-fold QLIKE and MSE(log) of the h-day vol forecast, 2011-2018."""
    errs_q, errs_m = [], []
    for fit_end, ev_start, ev_end in FOLDS:
        s = make().fit(prices.loc[:fit_end])
        upto = prices.loc[:ev_end]
        f = s.forecast(upto).loc[ev_start:ev_end]
        truth = np.exp(future_log_vol(upto, s.horizon)).loc[ev_start:ev_end]
        ok = f.notna() & truth.notna()
        v_hat, v = f[ok] ** 2, truth[ok] ** 2
        errs_q.append(v / v_hat - np.log(v / v_hat) - 1)  # QLIKE (Patton 2011)
        errs_m.append((np.log(f[ok]) - np.log(truth[ok])) ** 2)
    return {"qlike": float(pd.concat(errs_q).mean()), "mse_log": float(pd.concat(errs_m).mean())}


# -------------------------------------------------------------------- grids ---
def grid_A():
    for lags in (5, 10, 20):
        for hidden in ((8,), (32, 16)):
            for thr in (0.5, "base"):
                yield {"mode": "direction", "lags": lags, "hidden": hidden, "threshold": thr,
                       "l2": 1e-2, "band": 0.0, "scale": "none"}


def grid_C():
    for hidden in ((8,), (32, 16)):
        for tb in (5.0, 25.0):
            for l2 in (1e-3, 1e-1):
                for band in (0.0, 0.10):
                    yield {"mode": "sharpe", "hidden": hidden, "turnover_bps": tb, "l2": l2,
                           "band": band, "scale": "none"}


def grid_D():
    for model in ("har", "mlp"):
        for horizon in (5, 21):
            hs = ((16,), (32, 16)) if model == "mlp" else ((0,),)
            for hidden in hs:
                yield {"model": model, "horizon": horizon, "hidden": hidden}


def run_family(label, grid, make_fn, prices, log):
    rows = []
    for cfg in grid:
        t0 = time.time()
        wf = walk_forward(lambda c=cfg: make_fn(c), prices)
        row = {"family": label, **{k: str(v) for k, v in cfg.items()},
               **{k: v for k, v in wf.items() if not k.startswith("_")},
               "secs": round(time.time() - t0, 1)}
        rows.append(row)
        log(f"  {label} {cfg} -> WF Sharpe {wf['sharpe']:.2f}  exp {wf['exposure']:.2f}  "
            f"chg {wf['changes']}  ({row['secs']}s)")
    return pd.DataFrame(rows)


def seed_dispersion(cfg, prices, n=5) -> dict:
    """Walk-forward Sharpe of each seed alone (n_seeds=1) -> mean, sd."""
    s = [walk_forward(lambda k=k: NeuralNetStrategy(**{**cfg, "n_seeds": 1,
                                                      "seed": config.RANDOM_SEED + k}),
                      prices)["sharpe"] for k in range(n)]
    return {"seed_mean": float(np.mean(s)), "seed_sd": float(np.std(s, ddof=1)),
            "seeds": [round(x, 3) for x in s]}


# ---------------------------------------------------------------------- main ---
def main(with_lstm: bool, validation_only: bool = False) -> None:
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    prices = load_prices(source="csv", end=DEV_END)
    assert prices.index.max() <= pd.Timestamp(DEV_END), "test data must never be loaded"
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    if validation_only:                                  # reuse the saved train-only stage
        best = pd.read_csv(OUT / "best_per_family.csv", dtype=str)
        best["sharpe"] = best["sharpe"].astype(float)
        return run_validation(best, prices, log, lines)

    bh = walk_forward(BuyAndHold, prices)
    log(f"Buy & hold walk-forward (2011-2018) Sharpe {bh['sharpe']:.2f}")

    tables = [run_family("A", grid_A(), lambda c: NeuralNetStrategy(**c), prices, log),
              run_family("C", grid_C(), lambda c: NeuralNetStrategy(**c), prices, log),
              run_family("D", grid_D(), lambda c: VolForecastSizing(
                  **{k: v for k, v in c.items() if not (k == "hidden" and v == (0,))}),
                  prices, log)]
    for base in ("rv20", "ewma"):
        tables.append(run_family("D-baseline", [{"model": base, "horizon": 5}],
                                 lambda c: VolForecastSizing(**c), prices, log))
    if with_lstm:
        from nn_lstm import grid_lstm_direction, grid_lstm_sharpe, make_lstm  # noqa: E402
        tables.append(run_family("B", grid_lstm_direction(), make_lstm, prices, log))
        tables.append(run_family("C-LSTM", grid_lstm_sharpe(), make_lstm, prices, log))
    wf = pd.concat(tables, ignore_index=True)
    wf.to_csv(OUT / "walk_forward.csv", index=False)

    # Vol-forecast quality for family D (and its baselines)
    vq = {}
    baselines = [{"model": m, "horizon": h} for h in (5, 21) for m in ("rv20", "ewma")]
    for cfg in list(grid_D()) + baselines:
        c = {k: v for k, v in cfg.items() if not (k == "hidden" and v == (0,))}
        vq[json.dumps({k: str(v) for k, v in c.items()})] = vol_forecast_quality(
            lambda c=c: VolForecastSizing(**c), prices)
    pd.DataFrame(vq).T.to_csv(OUT / "vol_forecast_quality.csv")

    # Best configuration per family (by walk-forward Sharpe, train only)
    best = wf.loc[wf.groupby("family")["sharpe"].idxmax()]
    best.to_csv(OUT / "best_per_family.csv", index=False)
    log("\nBest per family (walk-forward 2011-2018, train only):\n"
        + best.drop(columns=["secs"]).to_string(index=False))

    # Seed dispersion for the NN families' picks
    disp = {}
    for fam in ("A", "C"):
        row = best[best.family == fam].iloc[0]
        cfg = _cfg_from_row(row)
        disp[fam] = seed_dispersion(cfg, prices)
        log(f"Seed dispersion {fam}: {disp[fam]}")
    (OUT / "seed_dispersion.json").write_text(json.dumps(disp, indent=2))

    return run_validation(best, prices, log, lines)


def run_validation(best, prices, log, lines) -> None:
    # Validation, looked at ONCE per family pick (plus buy & hold)
    val_rows = {"buy_and_hold": validation(BuyAndHold, prices)}
    for _, row in best.iterrows():
        cfg = _cfg_from_row(row)

        def make(c=cfg, fam=row.family):
            if fam in ("A", "C"):
                return NeuralNetStrategy(**c)
            if fam.startswith("D"):
                return VolForecastSizing(**c)
            from nn_lstm import make_lstm  # noqa: E402
            return make_lstm(c)

        val_rows[f"{row.family}"] = validation(make, prices)
    summary = {}
    for k, v in val_rows.items():
        summary[k] = {
            "train_sharpe": v["train"]["sharpe_ratio"],
            "val_sharpe": v["validation"]["sharpe_ratio"],
            "val_ci": [round(x, 2) for x in v["validation_ci"]],
            "val_cagr": v["validation"]["annualized_return"],
            "val_vol": v["validation"]["annualized_volatility"],
            "val_max_dd": v["validation"]["max_drawdown"],
            "val_exposure": v["validation"]["exposure"],
            "val_changes": v["validation"]["n_position_changes"],
            "train_max_dd": v["train"]["max_drawdown"],
        }
    # Classification quality of family A on validation
    a_row = best[best.family == "A"].iloc[0]
    a_strat = val_rows["A"]["_strategy"]
    p_up = a_strat.raw_output(prices).loc[VALID[0]:VALID[1]].to_numpy()
    fwd = F.forward_return(prices, 1).loc[VALID[0]:VALID[1]]
    y = np.where(fwd.notna(), (fwd > 0).astype(float), np.nan)  # last day has no target here
    summary["A"]["val_auc"] = auc(y, p_up)
    summary["A"]["val_share_long"] = float(np.nanmean(p_up > a_strat.threshold_))
    pd.DataFrame(summary).T.to_csv(OUT / "validation_once.csv")
    log("\nValidation (looked at once):\n" + pd.DataFrame(summary).T.round(3).to_string())

    # Post-hoc addendum (PROTOCOL.md): family C's pick with vol rescaling (v2b)
    c_cfg = {**_cfg_from_row(best[best.family == "C"].iloc[0]), "scale": "vol"}
    wf = walk_forward(lambda: NeuralNetStrategy(**c_cfg), prices)
    c_sharpe = float(best[best.family == "C"].iloc[0]["sharpe"])
    adopt = wf["sharpe"] >= c_sharpe - 0.05
    v = validation(lambda: NeuralNetStrategy(**c_cfg), prices)
    row = {"config": json.dumps({k: str(x) for k, x in c_cfg.items()}),
           "wf_sharpe": wf["sharpe"], "wf_exposure": wf["exposure"], "wf_changes": wf["changes"],
           "wf_max_dd": wf["max_dd"], "adopted_by_rule": bool(adopt),
           "scale_k": v["_strategy"].scale_,
           "train_sharpe": v["train"]["sharpe_ratio"],
           "train_cagr": v["train"]["annualized_return"],
           "train_max_dd": v["train"]["max_drawdown"], "train_exposure": v["train"]["exposure"],
           "val_sharpe": v["validation"]["sharpe_ratio"],
           "val_ci": [round(x, 2) for x in v["validation_ci"]],
           "val_cagr": v["validation"]["annualized_return"],
           "val_vol": v["validation"]["annualized_volatility"],
           "val_max_dd": v["validation"]["max_drawdown"],
           "val_exposure": v["validation"]["exposure"],
           "val_changes": v["validation"]["n_position_changes"]}
    pd.DataFrame({"C_v2b_scaled (post-hoc)": row}).T.to_csv(OUT / "v2b_scaled_posthoc.csv")
    log(f"\nv2b (post-hoc rescaling): {row}")
    (OUT / "log.txt").write_text("\n".join(lines) + "\n")
    _ = a_row


def _cfg_from_row(row) -> dict:
    skip = {"family", "sharpe", "cagr", "vol", "max_dd", "exposure", "changes", "secs"}
    cfg = {}
    for k, v in row.items():
        if k in skip or (isinstance(v, float) and np.isnan(v)) or v == "nan":
            continue
        try:
            cfg[k] = json.loads(v.replace("(", "[").replace(")", "]").replace("'", '"')
                                .replace(",]", "]"))
        except (json.JSONDecodeError, AttributeError):
            cfg[k] = v
        if k == "hidden":
            cfg[k] = tuple(cfg[k]) if isinstance(cfg[k], list) else (cfg[k],)
    if cfg.get("hidden") == (0,):
        cfg.pop("hidden")
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm", action="store_true")
    ap.add_argument("--validation-only", action="store_true")
    a = ap.parse_args()
    main(a.lstm, a.validation_only)
