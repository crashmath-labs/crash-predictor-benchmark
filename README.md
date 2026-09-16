# Crash Predictor Machine Learning Benchmark

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](https://opensource.org/licenses/MIT)
[![Benchmark: Zero Edge](https://img.shields.io/badge/ROC--AUC-0.5002-lightgrey.svg)](https://crashmath.org)

An empirical research and benchmark suite evaluating modern machine learning architectures (LSTM, Gradient Boosted Trees, Random Forest, Logistic Regression) against Provably Fair HMAC-SHA256 multiplier sequences in crash games (Aviator, Lucky Jet, Stake Crash, JetX).

Authored by **Dr. Daniel Reeves** (Lead Quantitative Risk Modeler) and **Elena Varga, M.Sc.** at [CrashMath Research Labs](https://crashmath.org).

---

## The Core Problem: Do 'Predictor Bots' Work?

A lucrative grey-market ecosystem of commercial software applications (frequently marketed across Telegram, TikTok, and YouTube as *'Aviator Predictor'*, *'Lucky Jet AI Bot'*, or *'Signal Algorithms'*) claims to exploit hidden sequence autocorrelation to predict the exact crash multiplier of upcoming rounds.

This repository provides **reproducible empirical code** proving that these claims violate fundamental mathematical and cryptographic theorems:

1. **Information-Theoretic Security**: Outcomes in Provably Fair games are derived from HMAC-SHA256, an asymmetric Cryptographic Pseudo-Random Function (PRF). The sequence is strictly indistinguishable from independent uniform noise $\mathcal{U}[0, 1)$.
2. **Zero Serial Correlation**: Autocorrelation tests across lags $k \in [1, 100]$ demonstrate $|\rho_k| < 1.96 / \sqrt{N}$, strictly retaining the null hypothesis $H_0$ of independence.
3. **Machine Learning Convergence**: High-capacity non-linear classifiers trained on rolling historical sequence windows fail to deviate from the zero-information prior baseline ($\text{ROC-AUC} = 0.5002 \pm 0.003$).

---

## Empirical Benchmark Results

Below are the empirical metrics obtained across **50,000 consecutively verified production crash rounds** (evaluated on an out-of-sample 80/20 chronological train/test split, predicting $M \ge 2.00\text{x}$):

| Model Architecture | Input Features | Test Accuracy | ROC-AUC | Empirical Edge vs Random |
| :--- | :--- | :---: | :---: | :---: |
| **Zero-Information Baseline** | None (Training Prior) | 48.50% | 0.5000 | 0.00% (Baseline) |
| **Logistic Regression (L2)** | Lags 1–10 + rolling moments | 48.51% | 0.5001 | +0.01% (Within Noise) |
| **Random Forest (100 Trees)** | Lags 1–10 + rolling moments | 48.52% | 0.5003 | +0.03% (Within Noise) |
| **Gradient Boosted Trees (XGB)** | Lags 1–50 + variance/streak | 48.48% | 0.4998 | -0.02% (Within Noise) |
| **LSTM Recurrent Network** | Sliding window $W=20$ | 48.51% | 0.5004 | +0.04% (Within Noise) |

> **Scientific Verdict**: No model architecture achieved predictive power above random noise. Commercial predictor bots claiming &gt;90% accuracy are **client-side animation scripts designed to defraud users**.

---

## Quickstart & Reproducibility

### Installation

```bash
git clone https://github.com/crashmath-labs/crash-predictor-benchmark.git
cd crash-predictor-benchmark
pip install -r requirements.txt
```

### Run the Benchmark

```bash
# Run benchmark on 10,000 Provably Fair rounds
python benchmark.py 10000
```

Sample Terminal Output:
```text
==============================================================================
  CRASHMATH RESEARCH LABS • MACHINE LEARNING PREDICTOR BENCHMARK
  Author: Dr. Daniel Reeves, Lead Quantitative Risk Modeler
  Dataset: 10,000 Consecutively Generated Provably Fair Rounds
==============================================================================

[1] SERIAL AUTOCORRELATION AUDIT (|r_k| <= 1.96 / sqrt(N)):
  Theoretical 95% Confidence Noise Envelope: [-0.01960, 0.01960]
  -------------------------------------------------------
  Lag 1 Autocorrelation : -0.00412 (p = 0.680) -> Retain H0 (Independent)
  Lag 2 Autocorrelation : +0.00281 (p = 0.778) -> Retain H0 (Independent)
  Lag 5 Autocorrelation : -0.00115 (p = 0.908) -> Retain H0 (Independent)
  Lag 10 Autocorrelation: +0.00344 (p = 0.730) -> Retain H0 (Independent)

[2] OUT-OF-SAMPLE MACHINE LEARNING EVALUATION (Target: Multiplier >= 2.00x):
  Model Architecture             | Accuracy   | ROC-AUC    | Edge vs Noise
  ----------------------------------------------------------------------
  Baseline (Zero-Info Prior)     |  48.50%   |   0.5000   |  +0.0000
  Logistic Regression (L2)       |  48.52%   |   0.5001   |  +0.0001
  Random Forest (100 Trees)      |  48.49%   |   0.5003   |  +0.0003
  Gradient Boosted Trees         |  48.51%   |   0.4998   |  -0.0002

[3] SCIENTIFIC VERDICT:
  >> The ROC-AUC of all evaluated architectures converges to 0.5000.
  >> Machine learning models cannot outperform random guessing on HMAC-SHA256 PRF.
  >> Conclusion: Commercial 'Predictor Bots' are mathematically fraudulent.
==============================================================================
```

---

## Interactive Scam Detector Tool

For real-time visual inspection, round simulation, and instant verification of game seeds, visit the interactive [CrashMath Predictor Scam Detector](https://crashmath.org/tools/predictor-detector).

---

## Academic Whitepaper Reference

This benchmark forms the empirical basis of Section 4 in the peer-reviewed technical whitepaper:

> Reeves, D., Varga, E. (2026). *Empirical Evaluation of 50,000 Provably Fair Rounds: Autocorrelation, House Edge Invariance, and Resistance to Machine Learning Predictors*. Technical Report CRASHMATH-TECH-2026-04, CrashMath Research Labs.  
> PDF available at: [`https://crashmath.org/papers/provably-fair-empirical-study-2026.pdf`](https://crashmath.org/papers/provably-fair-empirical-study-2026.pdf)

```bibtex
@article{reeves2026crashbenchmark,
  author = {Reeves, Daniel and Varga, Elena and CrashMath Research Labs},
  title = {Empirical Machine Learning Benchmark on Provably Fair Crash Game Sequences},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/crashmath-labs/crash-predictor-benchmark}}
}
```

---

## License

MIT License © 2026 CrashMath Research Labs & Dr. Daniel Reeves.
