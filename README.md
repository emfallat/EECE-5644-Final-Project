# EECE-5644 Final Project - Automatic Fall Detection (FallAllD)

Binary classification of **falls vs. activities of daily living (ADL)** from wearable
inertial sensors (FallAllD dataset: 15 subjects, 3 body positions, 6,605 recordings).
Team: Elizabeth Fallat, Darren Easler, Majd Khalaf.

### `FallDetection_ML.ipynb`
Preprocessing → feature engineering → 6 models → tuning → evaluation → feature
importance → research-question ablations → discussion. Models: Logistic Regression,
MLP, 1D-CNN, and LSTM in PyTorch, plus Random Forest and XGBoost (tree ensembles on
~30 engineered features). Structured to map onto the Iteration 4 task list (Parts 1 to 7).

**result:** XGBoost on ~30 accelerometer-focused
features - accuracy 0.961, recall 0.962, F1 0.941 (0.952 tuned), ROC-AUC 0.993; Random
Forest close behind (F1 0.939). Both tree ensembles beat the neural models. Accelerometer
alone is the most informative modality (F1 0.957); the barometer is essentially uninformative.

## Repo layout
| File | Purpose |
|---|---|
| `Iteration_FallDetection_ML.ipynb` | **Main deliverable** — full pipeline (6 models) + discussion |
| `Iteration_2_Proposal.pdf` | Earlier project proposal |
| `features.csv` | 84 engineered statistical features per recording |
| `ActivityID2Str.m` | MATLAB helper — converts activity ID to text description |
| `Description.txt` | Notes on FallAllD dataset helper scripts |
| `MECKD/` | Cloned reference repo (published baseline — next iteration) |
| `FallAllD.h5` | Consolidated dataframe — gitignored, rebuilt by notebook if missing |
| `FallAllD__zip.zip` | Raw dataset archive — gitignored, too large for GitHub |
| `FallAllD/` | Raw `.dat` files — gitignored, too large for GitHub |

## Environment / how to run
```bash
python3.12 -m venv .venv_meckd
source .venv_meckd/bin/activate
pip install numpy pandas tables scipy scikit-learn matplotlib seaborn xgboost \
            torch torchvision thop jupyter
# then open and run the notebook top-to-bottom (uses Apple MPS if available)
jupyter notebook Iteration_FallDetection_ML.ipynb
```
The notebook rebuilds `FallAllD.h5` / `features.csv` automatically if they are missing.

## Research questions (answered in the notebook)
1. Which model classifies falls vs. ADLs best? → **XGBoost** (F1 0.952 tuned, AUC 0.993).
2. Most useful sensing modality? → **Accelerometer** (F1 0.957); barometer useless.
3. Feature-selection impact? → Helps both linear and non-linear models.
4. Does device placement help? → Small consistent lift; all 3 positions classify well.
5. Are the models sufficient? → Strong baseline (recall 0.93 on unseen subjects), with caveats.
