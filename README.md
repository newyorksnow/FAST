# FAST: LSTM terrorism forecasting with multi-source data

Code for the experiments in *A Comparative Modeling Study of Algorithmic Behavior under Dataset Variation for Terrorism Forecasting* (Kieckens).

Project website: [Meteorismo: safety forecast](https://sites.google.com/view/meteorismo-safety-forecast/home)

## Overview

Machine-learning approaches to terrorism forecasting almost always train on a single source, the Global Terrorism Database (GTD, more than 180,000 events from 1970 to 2017). This study asks a narrow question: **does training an LSTM on multi-source socioeconomic and political data improve forecasts compared with GTD-only data?**

Two identical multi-output LSTM models are compared, so that only the data differs:

| Model | Training data |
|---|---|
| **GTD-only** | Global Terrorism Database |
| **Multi-source (MSD)** | GTD merged with protest/democracy/HDI indicators, coups, one-sided violence, non-state conflict (UCDP), GDP per capita and GNI |

Each model predicts two targets:

- **Attack success**, as a binary classification.
- **Country of attack**, as a multi-class classification.

### Method

- Temporal 70 / 15 / 15 train / validation / test split
- Feature scaling, grouping by country, and sequences of length 5
- Class-weighted losses
- 10 seeded runs per model; paired t-tests and Wilcoxon tests compare the runs

### Main findings

- **Binary (success):** the multi-source model had slightly higher accuracy, precision and F1 and lower log loss. The GTD-only model had marginally higher recall and AUC-ROC.
- **Categorical (country):** the GTD-only model did better on top-5 accuracy and macro precision, recall and F1, although macro scores stayed modest for both.
- **Conclusion:** multi-source data helped binary prediction in this setup but not categorical prediction. Replicating the comparison with other architectures (bi-LSTMs, transformers, tree-based models) and other datasets is future work.

## Repository contents

The scripts are plain Python files without a `.py` extension.

| File | Description |
|---|---|
| `cleandata` | Builds the multi-source dataset by combining GTD with GNI, GDP per capita, UCDP, coups and the other sources |
| `lstm-gtd` | LSTM trained on GTD only (preprocessing, training, evaluation) |
| `lstm-multidata` | The same LSTM trained on the merged multi-source data |
| `GTD_MSD_comparison` | Evaluates differences between the two datasets and models |
| `p-tests` | Paired t-test and Wilcoxon test on the per-run metrics of the two models |

## Reproducing the experiments

```bash
pip install pandas numpy scikit-learn tensorflow scipy
```

1. Obtain GTD and the supplementary datasets listed above and place them in one workspace folder. The scripts currently read from `/home/user/Downloads/workspace/`; edit the paths at the top of each script.
2. Run `cleandata` to build the merged dataset.
3. Run `lstm-gtd` and `lstm-multidata` for each seed. Set `SEED` at the top of each script (the experiments used seeds 1 to 9 and 43). Seeds are fixed for Python, NumPy and TensorFlow, and deterministic ops are enabled.
4. Save the per-seed metrics as `GTDRES.csv` and `MSDRES.csv`, then run `p-tests`.

## Data sources

Not included in this repository; obtain them from their providers under their own terms.

- Global Terrorism Database (GTD), University of Maryland (START)
- UCDP non-state conflict data
- Coup data, GDP per capita and GNI country series
- Protest, democracy and HDI indicators; one-sided violence data

## Citation

If you use this code, please cite the study above.
