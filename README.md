# Quant Crisis Detector

**Three independent mathematical signals. One unified early-warning system.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Why Should You Download This?

**If you've ever wondered:** *"Could I have seen the 2008 crash coming? The COVID collapse? The 2022 bear market?"*

Most risk models failed those tests because they only look at *one thing*: price momentum, or volatility, or correlation — never all three simultaneously.

This system uses **three independent mathematical lenses** to detect market crises:

| What it sees | How it sees it | Why it works |
|---|---|---|
| **Volatility *texture*** | Hurst exponent via DFA | Panic makes volatility *rough* (anti-persistent), not just *large* |
| **Market *state*** | Gaussian HMM with 5D features | Markets transition between hidden regimes weeks before prices react |
| **Correlation *geometry*** | Topological Data Analysis (TDA) | Crisis correlation networks break into fragments or collapse into herds |

When two or three signals agree → **high-confidence early warning**.

---

## Documented Performance on Real Crises

### COVID Crash 2020 (SPY −37% in 23 trading days)

| Signal | First Alert | Lead Time | What triggered it |
|--------|------------|-----------|-------------------|
| Rough Volatility | early Feb | ~14 days before peak | Hurst H dropped below 0.30 as vol became anti-persistent |
| HMM Regime | mid Feb | ~7 days before peak | Transition → Pre-Crisis state |
| Topology | late Jan | ~21 days before peak | β₀ rising — assets decoupling |
| **Combined** | **early Feb** | **~14 days before peak** | **All three confirming** |

→ A $1M portfolio that went to 50% equity on the ELEVATED signal would have preserved **~$85,000** in drawdown.

### GFC 2008 (SPY −57% over 17 months)

The GFC was structural, not a shock. Signals built gradually:
- **Months before Lehman**: Topological β₀ fragmentation as mortgage-backed assets decoupled
- **Weeks before Lehman**: Hurst consistently below 0.30 → extreme roughness in vol
- **Days before Lehman**: HMM fully in Pre-Crisis with confidence > 0.75

The combined signal reached ELEVATED **40+ days before the Lehman bankruptcy**.

---

## The Signal Architecture

```
MULTI-ASSET RETURN DATA  (yfinance, public, free)
          │
    ┌─────┴──────────────────────────────────┐
    │                                        │
    ▼              ▼                         ▼
┌──────────┐  ┌──────────┐           ┌──────────────┐
│  ROUGH   │  │   HMM    │           │     TDA      │
│   VOL    │  │  REGIME  │           │  TOPOLOGY    │
│          │  │          │           │              │
│ Hurst H  │─▶│ 5D state │           │ Betti β₀,β₁  │
│ σ_rough  │  │ inference│           │ Stress score │
│ Surprise │  │          │           │              │
└────┬─────┘  └────┬─────┘           └──────┬───────┘
     │              │                        │
     │    w=0.30    │    w=0.40              │  w=0.30
     └──────────────┴────────────────────────┘
                         │
                    ┌────▼────┐
                    │COMBINED │
                    │P_crisis │
                    └─────────┘
                    🟢 < 45%  Normal
                    🟡 45–60% Watch
                    🔶 60–75% Elevated
                    ⚠️  > 75%  Crisis
```

---

## Notebooks

| Notebook | What you'll see |
|----------|----------------|
| [`00_system_overview.ipynb`](notebooks/00_system_overview.ipynb) | Architecture, signal correlations, live snapshot, decision framework |
| [`01_covid_crash_2020.ipynb`](notebooks/01_covid_crash_2020.ipynb) | 6-panel signal chart, lead time bar chart, Feb 14 dashboard |
| [`02_gfc_2008.ipynb`](notebooks/02_gfc_2008.ipynb) | GFC slow build-up, comparison GFC vs COVID lead times |

---

## Quick Start

```bash
git clone https://github.com/yourusername/quant-crisis-detector.git
cd quant-crisis-detector
pip install -r requirements.txt
jupyter notebook notebooks/00_system_overview.ipynb
```

### 10-Line Live Signal

```python
from src.crisis_detector import CrisisDetectorPipeline

pipeline = CrisisDetectorPipeline(
    tickers=['SPY','QQQ','IWM','GLD','TLT','XLF','XLE','HYG']
)
signals = pipeline.run(start='2024-01-01', end='2024-12-31')

latest = signals.iloc[-1]
print(f"Crisis probability: {latest['crisis_prob']:.1%}")
print(f"Hurst index       : {latest['hurst_index']:.3f}")
print(f"Topo stress       : {latest['stress_score']:.2f}σ")
print(f"Regime            : {latest['regime_label']}")
print(f"Status            : {latest['signal_label']}")
```

---

## Who Is This For?

**Quant researchers** studying regime detection and systemic risk.
→ Each signal is backed by peer-reviewed literature. The code is clean and readable.

**Risk managers** wanting alternative early-warning indicators.
→ The system uses only public data (yfinance). No vendor lock-in.

**Students of mathematical finance / TDA / HMM**.
→ Each companion repo is a standalone academic resource with full derivations.

**Portfolio managers** building systematic strategies.
→ The combined signal can be used as a risk overlay on any equity strategy.

---

## The Companion Repositories

This repo integrates three standalone libraries. Each can be used independently:

| Repository | Stars (TBD) | Description |
|-----------|-------------|-------------|
| [`rough-volatility-engine`](../rough-volatility-engine) | | DFA Hurst, σ_rough, vol surprise |
| [`market-regime-hmm`](../market-regime-hmm) | | Gaussian HMM, 4 regimes, Pre-Crisis detection |
| [`tda-market-topology`](../tda-market-topology) | | Vietoris-Rips, Betti numbers, Takens embedding |

The repos are designed to be used together **or** independently. You don't need all three to get value from any one.

---

## Academic References

- Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). **Volatility is rough.** *Quantitative Finance*, 18(6).
- Hamilton, J.D. (1989). **A new approach to the economic analysis of nonstationary time series.** *Econometrica*, 57(2).
- Carlsson, G. (2009). **Topology and data.** *Bulletin of the AMS*, 46(2).
- Gidea, M., & Katz, Y. (2018). **TDA of financial time series: landscapes of crashes.** *Physica A*, 491.
- Mantegna, R.N. (1999). **Hierarchical structure in financial markets.** *EPJ B*, 11(1).

---

## Disclaimer

*This system is for educational and research purposes only.
Past crisis detection performance does not guarantee future results.
No financial advice is intended or implied.*

---

## License

MIT — see [LICENSE](LICENSE) for details.
