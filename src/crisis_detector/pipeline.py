"""
CrisisDetectorPipeline
======================
Orchestrates the three-signal system over a rolling time series.
Produces a daily crisis probability time series suitable for backtesting.

Usage
-----
    pipeline = CrisisDetectorPipeline(tickers=['SPY','QQQ','GLD','TLT','XLF','HYG'])
    results  = pipeline.run(start='2019-01-01', end='2021-12-31')
    results.plot()
"""

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .signals import CombinedSignal, RegimeSignal, RoughVolSignal, TopologySignal

warnings.filterwarnings("ignore")


# ── Inline implementations (no inter-repo imports needed) ─────────────────────

def _hurst_dfa(returns: np.ndarray,
               min_scale: int = 4, max_scale: int = 40, n_scales: int = 20) -> float:
    n = len(returns)
    eff_max = min(max_scale, n // 4)
    if n < min_scale * 2 or eff_max < min_scale:
        return -1.0
    y = np.cumsum(returns - np.mean(returns))
    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(eff_max), n_scales, dtype=int))
    fluct, valid = [], []
    for s in scales:
        n_seg = n // s
        if n_seg < 2:
            continue
        vars_ = []
        for k in range(n_seg):
            seg = y[k*s:(k+1)*s]
            t   = np.arange(s, dtype=float)
            try:
                c = np.polyfit(t, seg, 1)
                vars_.append(np.var(seg - np.polyval(c, t)))
            except Exception:
                continue
        if vars_:
            fluct.append(np.sqrt(np.mean(vars_)))
            valid.append(s)
    if len(fluct) < 4:
        return -1.0
    ls = np.log10(np.array(valid, dtype=float))
    lf = np.log10(np.array(fluct) + 1e-12)
    try:
        H, _ = np.polyfit(ls, lf, 1)
    except Exception:
        return -1.0
    return float(np.clip(H, 0.05, 0.95))


def _mantegna_distance(returns: pd.DataFrame, halflife: int = 21) -> np.ndarray:
    try:
        cov = returns.ewm(halflife=halflife, min_periods=10).cov().iloc[-len(returns.columns):]
        cov = cov.values.astype(float)
    except Exception:
        cov = np.cov(returns.values.T)
    var = np.diag(cov)
    var = np.where(var > 1e-12, var, 1e-12)
    vol = np.sqrt(var)
    rho = cov / np.outer(vol, vol)
    rho = np.clip(np.nan_to_num((rho + rho.T) / 2, nan=0.0), -1.0, 1.0)
    np.fill_diagonal(rho, 1.0)
    D = np.sqrt(np.clip(2*(1-rho), 0.0, 4.0))
    np.fill_diagonal(D, 0.0)
    return D


def _betti_union_find(D: np.ndarray, eps: float):
    n = D.shape[0]
    adj = (D <= eps) & (D > 0)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry: parent[rx] = ry
    edges = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                union(i, j); edges += 1
    comps = len(set(find(i) for i in range(n)))
    return float(comps), float(max(0, edges - (n - comps)))


class CrisisDetectorPipeline:
    """
    End-to-end rolling crisis detection pipeline.

    Parameters
    ----------
    tickers : list
        Asset basket for topology analysis (8–15 diversified ETFs recommended).
    hurst_window : int
        Rolling window for Hurst DFA estimation (default 60 days).
    topo_window : int
        Rolling window for correlation topology (default 60 days).
    step : int
        Compute every N days (default 5 — weekly).
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        hurst_window: int = 60,
        topo_window: int  = 60,
        step: int         = 5,
    ):
        self.tickers      = tickers or ['SPY','QQQ','IWM','GLD','TLT','XLF','XLE','HYG']
        self.hurst_window = hurst_window
        self.topo_window  = topo_window
        self.step         = step

        # Rolling histories for z-scores
        self._b0_hist: list = []
        self._b1_hist: list = []

    def run(self, start: str, end: str) -> pd.DataFrame:
        """
        Run the pipeline over [start, end] and return a DataFrame with:
            date, crisis_prob, rough_vol_p, regime_p, topo_p,
            hurst_index, vol_surprise, betti_0, betti_1, stress_score,
            signal_label
        """
        print(f"Downloading data {start} -> {end}...")
        # Use strftime to produce clean date strings — avoids yfinance timestamp parsing errors
        start_buffer = (pd.Timestamp(start) - pd.Timedelta(days=120)).strftime('%Y-%m-%d')
        end_str      = pd.Timestamp(end).strftime('%Y-%m-%d')
        prices = yf.download(
            self.tickers, start=start_buffer, end=end_str, progress=False
        )['Close'].dropna(axis=1, how='all').dropna()

        spy_prices = yf.download('SPY', start=start, end=end_str,
                                  progress=False)['Close'].squeeze()

        returns    = np.log(prices / prices.shift(1)).dropna()
        spy_ret    = np.log(spy_prices / spy_prices.shift(1)).dropna()

        window = max(self.hurst_window, self.topo_window)
        records = []

        print(f"Computing signals (step={self.step}d, n_obs={len(returns)})...")
        for i in range(window, len(returns), self.step):
            date = returns.index[i - 1]
            win  = returns.iloc[i - window : i]
            ret_arr = win['SPY'].values if 'SPY' in win.columns else win.iloc[:, 0].values

            # ── Signal 1: Rough Volatility ──────────────────────────────────
            H = _hurst_dfa(ret_arr[-self.hurst_window:])
            H = H if H >= 0 else 0.45

            sigma_std   = np.std(ret_arr[-20:]) * np.sqrt(252)
            sigma_long  = np.std(ret_arr[-60:]) * np.sqrt(252)
            frac_scale  = (20/60) ** (H - 0.5)
            sigma_fcast = sigma_long * frac_scale
            surprise    = (sigma_std - sigma_fcast) / (sigma_fcast + 1e-8)
            surprise    = float(np.clip(surprise, -2.0, 5.0))
            sigma_rough = sigma_std * (1.0 + 2.0 * (0.5 - H))

            rv_sig = RoughVolSignal(
                hurst_index=H, sigma_rough=sigma_rough,
                sigma_standard=sigma_std, vol_surprise=surprise,
                vol_regime_change_p=float(np.clip(2*(0.5-H) + abs(surprise)/4, 0, 1))
            )

            # ── Signal 2: Regime (heuristic — HMM is slow for rolling) ─────
            # Simple rule-based proxy: high vol + negative trend → Bear
            trend_20  = float(np.sum(ret_arr[-20:]))
            vol_z     = (sigma_std - sigma_long) / (sigma_long + 1e-8)
            if H < 0.30 and surprise > 0.8:
                regime_lbl, conf = "Pre-Crisis", 0.75
            elif trend_20 < -0.03 and vol_z > 0.3:
                regime_lbl, conf = "Bear / High Vol", 0.70
            elif trend_20 > 0.02 and vol_z < 0:
                regime_lbl, conf = "Bull / Low Vol", 0.70
            else:
                regime_lbl, conf = "Transition", 0.50

            re_sig = RegimeSignal(regime_label=regime_lbl, confidence=conf)

            # ── Signal 3: Topology ──────────────────────────────────────────
            try:
                D   = _mantegna_distance(win.tail(self.topo_window))
                tri = D[np.triu_indices(D.shape[0], k=1)]
                eps = float(np.percentile(tri, 40))

                try:
                    from ripser import ripser as _rip
                    res  = _rip(D, metric='precomputed', maxdim=1)
                    dgms = res['dgms']
                    fin0 = dgms[0][np.isfinite(dgms[0][:, 1])]
                    b0   = float(np.sum(fin0[:, 1] - fin0[:, 0])) if len(fin0) else 0.0
                    b1   = 0.0
                    if len(dgms) > 1:
                        fin1 = dgms[1][np.isfinite(dgms[1][:, 1])]
                        b1   = float(np.sum(fin1[:, 1] - fin1[:, 0])) if len(fin1) else 0.0
                except Exception:
                    b0, b1 = _betti_union_find(D, eps)
            except Exception:
                b0, b1 = 0.0, 0.0

            self._b0_hist.append(b0)
            self._b1_hist.append(b1)
            if len(self._b0_hist) > 252:
                self._b0_hist.pop(0)
                self._b1_hist.pop(0)

            stress = 0.0
            if len(self._b0_hist) >= 10:
                a0, a1 = np.array(self._b0_hist), np.array(self._b1_hist)
                if a0.std() > 1e-4 and a1.std() > 1e-4:
                    z0 = (b0 - a0.mean()) / a0.std()
                    z1 = (b1 - a1.mean()) / a1.std()
                    stress = float(0.4 * z0 + 0.6 * z1)

            stress_level = "alert" if stress > 2.5 else "elevated" if stress > 2.0 else "normal"
            tp_sig = TopologySignal(betti_0=b0, betti_1=b1,
                                    stress_score=stress, stress_level=stress_level)

            # ── Combined ────────────────────────────────────────────────────
            combined = CombinedSignal(rough_vol=rv_sig, regime=re_sig, topology=tp_sig)

            records.append({
                'date':          date,
                'crisis_prob':   combined.crisis_probability,
                'rough_vol_p':   rv_sig.crisis_probability,
                'regime_p':      re_sig.crisis_probability,
                'topo_p':        tp_sig.crisis_probability,
                'hurst_index':   H,
                'vol_surprise':  surprise,
                'betti_0':       b0,
                'betti_1':       b1,
                'stress_score':  stress,
                'signal_label':  combined.label,
                'regime_label':  regime_lbl,
            })

        if not records:
            print("Warning: no signal points computed. Check that data downloaded correctly.")
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index('date')
        df.index = pd.to_datetime(df.index)
        print(f"Done: {len(df)} signal points.")
        return df
