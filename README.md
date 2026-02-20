# California Housing Price Prediction — DNN & Gradient Boosting Regression

Predicting **median house values** for California districts using a **Deep Neural Network** and **Gradient Boosting Regressor**, trained on the [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) (20,640 samples). The notebook applies batch normalisation, dropout, learning rate scheduling, and stochastic boosting, with a full suite of regression diagnostics and geographic visualisation.

---

## Table of Contents

- [Task Overview](#task-overview)
- [Dataset](#dataset)
- [Feature Descriptions](#feature-descriptions)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures](#model-architectures)
  - [Model 1: Deep Neural Network (DNN)](#model-1-deep-neural-network-dnn)
  - [Model 2: Gradient Boosting Regressor](#model-2-gradient-boosting-regressor)
- [Optimisation Techniques](#optimisation-techniques)
- [Evaluation Metrics](#evaluation-metrics)
- [Prediction & Inference](#prediction--inference)
- [Getting Started](#getting-started)
- [Notebook Structure](#notebook-structure)
- [Outputs](#outputs)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Task Overview

```
┌───────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│ Neighbourhood Features│─────▶│  DNN / Gradient      │─────▶│  Median House Value  │
│ (income, age, rooms,  │      │  Boosting Regressor  │      │  (in $100,000s)      │
│  location, occupancy) │      └──────────────────────┘      └──────────────────────┘
└───────────────────────┘
```

- **Input:** 8 neighbourhood-level features (median income, house age, rooms, bedrooms, population, occupancy, latitude, longitude)
- **Output:** Median house value in units of $100,000 (continuous)

**Real-world applications:**
- Property valuation and price estimation
- Real estate market analysis
- Urban planning and housing policy research
- Mortgage risk assessment

---

## Dataset

**[California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)** — derived from the 1990 U.S. Census, built into scikit-learn.

| Property | Value |
|----------|-------|
| Samples | 20,640 block groups |
| Features | 8 |
| Target | `MedHouseVal` — median house value ($100,000s) |
| Source | `sklearn.datasets.fetch_california_housing` |
| Geography | California, USA |
| Granularity | Census block groups (600–3,000 people) |

No manual download required — the dataset is loaded directly from scikit-learn.

---

## Feature Descriptions

| Feature | Description | Type |
|---------|-------------|------|
| `MedInc` | Median income in block group (tens of thousands $) | Continuous |
| `HouseAge` | Median house age in block group (years) | Continuous |
| `AveRooms` | Average number of rooms per household | Continuous |
| `AveBedrms` | Average number of bedrooms per household | Continuous |
| `Population` | Block group population | Continuous |
| `AveOccup` | Average household occupancy (members per household) | Continuous |
| `Latitude` | Block group latitude coordinate | Continuous |
| `Longitude` | Block group longitude coordinate | Continuous |
| **`MedHouseVal`** | **Median house value (TARGET, in $100,000s)** | **Continuous** |

---

## Data Preprocessing

1. **Missing value check** — confirmed zero missing values
2. **Train/test split** — 80/20 with `random_state=42`
3. **Feature scaling** — `StandardScaler` (zero mean, unit variance) — essential for DNN convergence

---

## Model Architectures

### Model 1: Deep Neural Network (DNN)

```
Input (8 features)
    │
Dense(256, ReLU) → BatchNorm → Dropout(0.3)
    │
Dense(128, ReLU) → BatchNorm → Dropout(0.3)
    │
Dense(64, ReLU)  → BatchNorm → Dropout(0.2)
    │
Dense(32, ReLU)  → BatchNorm
    │
Dense(1, Linear) → Output (house value)
```

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam (LR = 1e-3) |
| Loss function | MSE |
| Batch size | 64 |
| Epochs | 100 (with early stopping) |
| Validation split | 20% of training data |
| Early stopping | Patience 15, restore best weights |
| LR scheduler | `ReduceLROnPlateau` — factor 0.5, patience 7, min LR 1e-6 |

### Model 2: Gradient Boosting Regressor

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `max_depth` | 5 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `min_samples_split` | 5 |
| `min_samples_leaf` | 2 |

A sequential ensemble of 200 shallow decision trees, each fitted on the residuals of the previous round. Stochastic subsampling (80%) adds regularisation and reduces overfitting.

---

## Optimisation Techniques

### DNN

| # | Technique | Configuration | Purpose |
|---|-----------|---------------|---------|
| 1 | **Batch Normalisation** | After each dense layer | Stabilises training, enables higher learning rates |
| 2 | **Dropout** | 30% / 30% / 20% across hidden layers | Prevents overfitting |
| 3 | **Early Stopping** | Patience 15 on `val_loss`, restore best weights | Prevents overfitting, saves time |
| 4 | **Learning Rate Scheduling** | Halves LR when `val_loss` plateaus (patience 7) | Fine-grained optimisation near convergence |
| 5 | **Feature Scaling** | `StandardScaler` | Essential for neural network convergence |

### Gradient Boosting

| # | Technique | Configuration | Purpose |
|---|-----------|---------------|---------|
| 1 | **Step-size shrinkage** | `learning_rate=0.1` | Controls contribution of each tree |
| 2 | **Tree depth limiting** | `max_depth=5` | Prevents individual trees from overfitting |
| 3 | **Stochastic subsampling** | `subsample=0.8` | Reduces variance via random 80% data sampling |
| 4 | **Node regularisation** | `min_samples_split=5`, `min_samples_leaf=2` | Prevents leaf nodes with too few samples |

---

## Evaluation Metrics

Both models are evaluated on the same 20% held-out test set:

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error — average squared difference |
| **RMSE** | Root Mean Squared Error — interpretable in $100,000s |
| **R² Score** | Coefficient of determination (1.0 = perfect) |
| **MAPE** | Mean Absolute Percentage Error |

### Visualisations

The notebook produces:

- **Target distribution** — histogram and box plot of median house values
- **Correlation heatmap** — pairwise Pearson correlation of all features + target
- **Geographic price map** — scatter plot of house values by latitude/longitude (California map)
- **DNN training curves** — loss and MAE over epochs
- **Predicted vs Actual** — scatter plots for both models with perfect-prediction reference line
- **Residual analysis** — residual scatter plots and residual distribution histograms for both models
- **Feature importance** — Gradient Boosting feature importance ranking
- **Model comparison** — side-by-side bar chart of all metrics

---

## Prediction & Inference

The notebook includes a reusable function for predicting house prices from neighbourhood characteristics:

```python
predicted_value = predict_house_price(
    neighborhood_data={
        'MedInc': 8.5,        # High income area
        'HouseAge': 15,       # Newer homes
        'AveRooms': 7.0,      # Spacious
        'AveBedrms': 1.1,
        'Population': 1200,
        'AveOccup': 2.5,
        'Latitude': 34.05,    # Near coast (LA area)
        'Longitude': -118.25,
    },
    model=gb_regressor,
    scaler=scaler,
    feature_names=X_housing.columns,
)
# Output: PREDICTED HOUSE VALUE: $XXX,XXX
```

Three example neighbourhoods are demonstrated:
1. **Luxury** — high income, coastal, newer homes
2. **Budget** — lower income, inland, older homes
3. **Mid-range** — average income, suburban

---

## Getting Started

### Requirements

- **Hardware:** GPU recommended for DNN training (Google Colab or local CUDA GPU)
- **Python:** 3.8+

### Installation

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

### Running

1. Open `Regression__DNN_and_Gradient_Boost_.ipynb` in Jupyter or Google Colab.
2. Run all cells sequentially.
3. The California Housing dataset is loaded automatically from scikit-learn — no manual download required.

---

## Notebook Structure

| Cell(s) | Section | Description |
|---------|---------|-------------|
| 0–1 | Introduction | Task overview, dataset info |
| 2 | Setup | Import libraries (TF/Keras, sklearn, visualisation) |
| 3–4 | Data Loading | Load California Housing from sklearn, create DataFrame |
| 5–8 | EDA | Sample rows, feature descriptions, statistics, target distribution |
| 9–11 | Visualisation | Target histogram/boxplot, correlation heatmap, geographic price map |
| 12–15 | Preprocessing | Missing value check, 80/20 split, `StandardScaler` |
| 16–17 | Evaluation Setup | `evaluate_regression()` function (MSE, RMSE, R², MAPE) |
| 18–23 | Model 1: DNN | Build 4-layer architecture, compile, train (100 epochs), training curves, evaluate |
| 24–26 | Model 2: Gradient Boosting | Train with tuned hyperparameters, evaluate |
| 27–33 | Results | Summary table, predicted vs actual plots, residual analysis, feature importance, model comparison bar chart |
| 34–35 | Final Summary | Complete results and optimisation techniques recap |
| 36–41 | Inference | `predict_house_price()` function, luxury/budget/mid-range examples, sample predictions table |

---

## Outputs

| Artifact | Description |
|----------|-------------|
| Results comparison table | Side-by-side DNN vs Gradient Boosting metrics |
| DNN training curves | Loss and MAE across epochs |
| Predicted vs Actual plots | Scatter plots for both models |
| Residual analysis | Scatter plots and histograms for both models |
| Geographic price map | California house values by lat/long |
| Feature importance chart | Gradient Boosting feature ranking |

---

## Tech Stack

| Library | Role |
|---------|------|
| [TensorFlow / Keras](https://www.tensorflow.org/) | Deep Neural Network architecture and training |
| [scikit-learn](https://scikit-learn.org/) | Gradient Boosting, preprocessing, evaluation metrics, dataset loading |
| [Matplotlib / Seaborn](https://matplotlib.org/) | EDA, training curves, residual analysis, geographic plots |
| [NumPy / Pandas](https://numpy.org/) | Data manipulation |

---

## License

This project is for educational and research purposes.
