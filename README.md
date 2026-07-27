# EECE-5644 Final Project - Automatic Fall Detection (FallAllD)

Binary classification of **falls vs. activities of daily living (ADL)** from wearable
inertial sensors (FallAllD dataset: 15 subjects, 3 body positions, 6,605 recordings).
Team: Elizabeth Fallat, Darren Easler, Majd Khalaf.

## ML iteration deliverable
Everything for this iteration is in one notebook, **all models in PyTorch**:

### `Iteration_FallDetection_ML.ipynb`
Preprocessing → feature engineering → 6 models → tuning → evaluation → feature
importance → research-question ablations → discussion. Models: Logistic Regression,
MLP, 1D-CNN, and LSTM in PyTorch, plus Random Forest and XGBoost (tree ensembles on
~30 simple features). Structured to map onto the Iteration 4 task list (Parts 1 to 7).

### `Iteration4_Report.docx`
The submission report: algorithm justification, train/test process, preprocessing,
tuning results, metrics, all required visualizations, discussion, and research-question
answers.

**Headline result (held-out test subjects):** XGBoost on ~30 accelerometer-focused
features - accuracy 0.961, recall 0.962, F1 0.941 (0.952 tuned), ROC-AUC 0.993; Random
Forest close behind (F1 0.939). Both tree ensembles beat the neural models. Accelerometer
alone is the most informative modality (F1 0.957); the barometer is essentially uninformative.

## Repo layout
| File | Purpose |
|---|---|
| `Iteration_FallDetection_ML.ipynb` | **Main deliverable** - full pipeline (6 models) + report/discussion (builds the data itself) |
| `Iteration4_Report.docx` | **Submission report** (Word) generated from the notebook results |
| `FallAllD.h5` | Consolidated dataframe (6,605 recordings; Acc/Gyr/Mag/Bar + labels) |
| `features.csv` | 84 engineered statistical features per recording |
| `figures/` | Generated plots (EDA, confusion matrices, ROC, importance, ablations) |
| `results/` | Metric tables (CSV) for every experiment |
| `FallAllD__zip.zip` | Raw dataset (not tracked if large) |
| `MECKD/` | Cloned reference repo (published baseline - next iteration) |

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
1. Which model classifies falls vs. ADLs best? → **MLP** (F1 0.917, AUC 0.983).
2. Most useful sensing modality? → **Accelerometer** (F1 0.957); barometer useless.
3. Feature-selection impact? → Helps both linear and non-linear models.
4. Does device placement help? → Small consistent lift; all 3 positions classify well.
5. Are the models sufficient? → Strong baseline (recall 0.93 on unseen subjects), with caveats.

## Next iteration (professor's brief)
Reproduce the **MECKD** IEEE Sensors 2024 baseline (ResNet/CNN/MobileNet + knowledge
distillation) and compare classical vs. deep models at **5 lead times** (predicting a
fall before impact). Scaffolding (`MECKD/`, environment, data pipeline) is already in place.
