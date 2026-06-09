# AI Crypto Forecaster (Transformer)

## English below

Zastosowanie głębokiego uczenia sekwencyjnego (architektura **Transformer**) do wielokrokowej predykcji trendów i generowania sygnałów transakcyjnych na rynku kryptowalut (BTC/USDT).

---

## Opis projektu

System realizuje **hybrydowy model głębokiego uczenia** typu Multi-task, który na podstawie 168 godzin historii rynku jednocześnie:

- **prognozuje cenę** BTC/USDT na 24 godziny do przodu (zadanie regresji),
- **generuje sygnał transakcyjny** BUY / SELL / HOLD (zadanie klasyfikacji).

Model oparty jest na enkoderze Transformera z mechanizmem **Multi-Head Self-Attention** i dwiema głowicami wyjściowymi współdzielącymi wspólną reprezentację (*hard parameter sharing*).

---

## Główne funkcjonalności

- **Potok danych** — automatyczne pobieranie świec BTC/USDT (1h) z giełdy KuCoin (`ccxt`), czyszczenie i inżynieria cech.
- **Inżynieria cech** — OHLCV + Log-Returns, SMA_24, SMA_168, RSI, ATR.
- **Architektura Transformer** w PyTorch — projekcja wejścia, kodowanie pozycyjne, enkoder z Self-Attention, dwie głowice (regresja + klasyfikacja).
- **Trening Multi-task** — łączna funkcja straty `L = α·MSE + β·CrossEntropy`, AdamW, Early Stopping, ReduceLROnPlateau, gradient clipping, ważenie klas.
- **Ewaluacja** — metryki regresji (MAE, RMSE, MAPE) oraz klasyfikacji (Precision, Recall, F1).
- **Backtesting** — symulacja strategii transakcyjnej vs. Buy & Hold.
- **Grid Search** — optymalizacja hiperparametrów (d_model, num_layers, dropout, lr).
- **Analiza okna wejściowego** — wpływ długości historii (48h / 96h / 168h / 336h).
- **Interfejs Gradio** — interaktywna wizualizacja predykcji i decyzji modelu.

---

## Stos technologiczny

| Kategoria | Biblioteki |
|---|---|
| Deep Learning | PyTorch |
| Dane i obliczenia | Pandas, NumPy |
| ML / preprocessing | Scikit-learn |
| Dane giełdowe | ccxt |
| Wizualizacja | Matplotlib, Seaborn |
| Interfejs | Gradio |

---

## Instalacja

Wymagany **Python 3.10+**.

```bash
# 1. Sklonuj repozytorium
git clone https://github.com/[TWOJA-NAZWA]/[NAZWA-REPO].git
cd [NAZWA-REPO]

# 2. (Zalecane) utwórz i aktywuj środowisko wirtualne
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# 3. Zainstaluj zależności
pip install -r requirements.txt
```

---

## Uruchomienie

```bash
jupyter notebook crypto_training.ipynb
```

Notatnik można też uruchomić w **JupyterLab** lub **Google Colab**. Komórki wykonuje się kolejno (Krok 1 → Krok 8). Ostatni krok uruchamia interaktywny interfejs Gradio z wizualizacją predykcji.

---

## Struktura projektu

```
.
├── crypto_training.ipynb   # Główny notatnik (pełna implementacja, Kroki 1-8)
├── requirements.txt        # Zależności Python
└── README.md               # Ten plik
```

---

## Etapy w notatniku

1. **Krok 1** — Pobieranie danych z giełdy i analiza statystyczna (rozkład klas, korelacje, Log-Returns).
2. **Krok 2** — Sliding Window + normalizacja (RobustScaler), podział chronologiczny 70/15/15.
3. **Krok 3** — Budowa architektury Transformer (enkoder + dwie głowice).
4. **Krok 4** — Trening modelu (Early Stopping, krzywe uczenia).
5. **Krok 5** — Ewaluacja: metryki regresji/klasyfikacji + backtesting.
6. **Krok 6** — Grid Search hiperparametrów.
7. **Krok 7** — Analiza wpływu długości okna wejściowego.
8. **Krok 8** — Interaktywny interfejs Gradio.

---

## Wyniki (skrót)

- Najlepsza konfiguracja Grid Search: `d_model=64, num_layers=1, dropout=0.3, lr=0.0001` → Test RMSE ≈ 13 139 USD.
- Najlepsza długość okna: **48h** (RMSE ≈ 16 209 USD) — krótsze okno przewyższa dłuższe na zmiennym rynku.
- Backtesting: strategia AI 0,00% vs. Buy & Hold +14,11% w badanym okresie wzrostowym.

---

## Zastrzeżenie

Projekt ma charakter **edukacyjno-badawczy** i **nie stanowi rekomendacji inwestycyjnej**. Handel kryptowalutami wiąże się z wysokim ryzykiem straty kapitału.

---
---

# AI Crypto Forecaster (Transformer) — English

Sequential deep learning (**Transformer** architecture) for multi-step trend prediction and trading-signal generation on the cryptocurrency market (BTC/USDT).

---

## Overview

The system implements a **hybrid multi-task deep learning model** that, based on 168 hours of market history, simultaneously:

- **forecasts the price** of BTC/USDT 24 hours ahead (regression task),
- **generates a trading signal** BUY / SELL / HOLD (classification task).

The model is built on a Transformer encoder with a **Multi-Head Self-Attention** mechanism and two output heads sharing a common representation (*hard parameter sharing*).

---

## Key Features

- **Data pipeline** — automatic download of BTC/USDT candles (1h) from the KuCoin exchange (`ccxt`), cleaning and feature engineering.
- **Feature engineering** — OHLCV + Log-Returns, SMA_24, SMA_168, RSI, ATR.
- **Transformer architecture** in PyTorch — input projection, positional encoding, Self-Attention encoder, two heads (regression + classification).
- **Multi-task training** — combined loss `L = α·MSE + β·CrossEntropy`, AdamW, Early Stopping, ReduceLROnPlateau, gradient clipping, class weighting.
- **Evaluation** — regression metrics (MAE, RMSE, MAPE) and classification metrics (Precision, Recall, F1).
- **Backtesting** — trading-strategy simulation vs. Buy & Hold.
- **Grid Search** — hyperparameter optimization (d_model, num_layers, dropout, lr).
- **Input-window analysis** — impact of history length (48h / 96h / 168h / 336h).
- **Gradio interface** — interactive visualization of predictions and model decisions.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Deep Learning | PyTorch |
| Data & computation | Pandas, NumPy |
| ML / preprocessing | Scikit-learn |
| Exchange data | ccxt |
| Visualization | Matplotlib, Seaborn |
| Interface | Gradio |

---

## Installation

Requires **Python 3.10+**.

```bash
# 1. Clone the repository
git clone https://github.com/[YOUR-USERNAME]/[REPO-NAME].git
cd [REPO-NAME]

# 2. (Recommended) create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
jupyter notebook crypto_training.ipynb
```

The notebook can also be run in **JupyterLab** or **Google Colab**. Run the cells sequentially (Step 1 → Step 8). The final step launches the interactive Gradio interface with prediction visualization.

---

## Project Structure

```
.
├── crypto_training.ipynb   # Main notebook (full implementation, Steps 1-8)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Notebook Steps

1. **Step 1** — Exchange data download and statistical analysis (class distribution, correlations, Log-Returns).
2. **Step 2** — Sliding Window + normalization (RobustScaler), chronological 70/15/15 split.
3. **Step 3** — Transformer architecture (encoder + two heads).
4. **Step 4** — Model training (Early Stopping, loss curves).
5. **Step 5** — Evaluation: regression/classification metrics + backtesting.
6. **Step 6** — Hyperparameter Grid Search.
7. **Step 7** — Input-window length impact analysis.
8. **Step 8** — Interactive Gradio interface.

---

## Results (summary)

- Best Grid Search configuration: `d_model=64, num_layers=1, dropout=0.3, lr=0.0001` → Test RMSE ≈ 13,139 USD.
- Best window length: **48h** (RMSE ≈ 16,209 USD) — a shorter window outperforms longer ones on a volatile market.
- Backtesting: AI strategy 0.00% vs. Buy & Hold +14.11% over the studied uptrend period.

---

## Disclaimer

This project is for **educational and research purposes** and **does not constitute investment advice**. Cryptocurrency trading carries a high risk of capital loss.
