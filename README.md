# NLP Coursework

## Maybe the Most Important Part for Assessors

I make this section for the assessors to find what they need to assess my work and review the code. Please read through this README.md.

- `dev.txt` and `test.txt` are under `BestModel/`.
- [Google Drive link](https://drive.google.com/drive/folders/1d4GaaNqLkzNArTVoM2qtZgoRT7sk3JdA?usp=sharing) has my model. In case you want to reproduce the model or evaluate on your own, maybe to verify `BestModel/dev.txt` and `BestModel/test.txt`, please look **Reproducing the Best Model (5-Fold LAN Ensemble)** section below.
- You can find the source code of all the training pipeline, including the model, under `src/`. For more details look **Codebase Buide for Assessors** section below.

These choices were discussed with and allowed by Dr. Chiraag Lala. I hope there are no disadvantages for such structural choices.

## Setup

This coursework uses `uv` for package management. `uv` is a Python package and environment manager from [Astral](https://astral.sh). It replaces tools like `pip`, `pipx`, `conda`, and `virtualenv` with a single, simple interface. It is also much faster than prior tools.

### Installing `uv`

Run the following in your terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, open a new terminal so `uv` is on your `PATH`.

### Always use `uv run`

Do **not** run `python` or `pip` directly. Always run scripts through `uv run` so dependencies
and environments are handled automatically. If you want to add a new dependency, you can use `uv add`. This will add the dependency to `pyproject.toml`, update `uv.lock`, and install the package into your virtual environment.

### If you have to use pip

You can install the dependencies using standard pip by `pip install -r requirements.txt`, but I strongly recommend using `uv sync`.

## Reproducing the Best Model (5-Fold LAN Ensemble)

The proposed architecture utilizes a 5-Fold Hard Voting Ensemble. To reproduce the evaluation results on the official Dev Set, please follow these steps:

### 1. Download the Model Checkpoints

Download the 5 model folds and the corresponding thresholds JSON from the [Google Drive link](https://drive.google.com/drive/folders/1d4GaaNqLkzNArTVoM2qtZgoRT7sk3JdA?usp=sharing).

### 2. File Placement

Place all downloaded files directly into the `BestModel/` directory at the root of this repository. Your folder structure should look exactly like this:
```text
nlp/
├── BestModel/
│   ├── fold_thresholds.json
│   ├── model_fold_1.pth
│   ├── model_fold_2.pth
│   ├── model_fold_3.pth
│   ├── model_fold_4.pth
│   └── model_fold_5.pth
├── data/
├── src/
│   └── evaluate_ensemble.py
├── eval/
└── README.md
```

### 3. Run the Evaluation Script

Ensure your `uv` have installed all the dependencies. Run the ensemble evaluation code from the root directory by `uv run python src/evaluate_ensemble.py` or via gpucluster in Imperial by `sbatch submit_ensemble.sh`.

The script will sequentially load each fold, apply the specific thresholds optimized during training, and output the final ensembled F1 score (~0.5289) along with a detailed classification report. In fact, I copy pasted the report, you can find it in `BestModel/results.out`.

## Codebase Guide for Assessors

To understand the lifecycle of the proposed model, the codebase is modularized into specific directories and scripts. If you are reviewing the code, please follow this logical execution pipeline:

### 1. Core Execution Pipeline (`src/`)

These scripts represent the primary lifecycle of the model, from hyperparameter tuning to final submission generation of `BestModel/dev.txt` and `BestModel/test.txt`. They are intended to be run in the following order.

1. `src/tune.py` (Hyperparameter Optimization): Executes Optuna sweep to find optimal hyperparameters using median pruning to terminate unpromising trials early.

2. `src/train_kfold.py` (Model Training): Trains 5-fold cross-validation ensemble. It loads optimal parameters from the first step, applies R-Drop regularization, isolates training data, and saves 5 best model folds alongside their optimized classification thresholds.

3. `src/evaluate_ensemble.py` (Inference and Evaluation): Loads 5 trained models and their respective thresholds, then runs inference on the official Dev set, and applies hard-voting mechanism to output the final F1 score (0.5289) and classification reports.

4. `src/generate_submission.py` (Final Output): Uses the trained 5-fold ensemble to generate the binary predictions for the official blind test set, outputting the required `BestModel/dev.txt` and `BestModel/test.txt` files for submission.

### 2. Architecture and Data Processing (`src/`)

1. `src/model.py`: Contains `PCLModelWithLAN`. This includes the `DeBERTa-v3-base` backbone, the custom Label Attention Network (LAN) using multi-head attention, and the Multi-Task Learning (MTL) taxonomy heads.

2. `src/data_loader.py`: Handles robust data ingestion. It parses the binary labels, correctly aggregates the multi-span, multi-label taxonomy categories from the PCL dataset, cleans the text, and ensures strict isolation between the training and validation splits.

### 3. Local Evaluation and Ablation Studies (`eval/`)

Codes under `eval/` directory were generated the analyses for Exercise 5.2 of the report.

1. `eval/compare_models.py`: Generates 4-bucket error analysis comparing the proposed model against the vanilla `cardiffnlp/pcl_robertabase` baseline.

2. `eval/run_ablation.py`: A self-contanied script that trains and evaluates a stripped-down version of the proposed model (removing LAN and MTL heads entirely).

3. `eval/compare_ablation.py`: Compares the predictions of the proposed model against the ablated model to isolate exactly which texts the LAN successfully filtered or falsely flagged.

4. `eval/count_dev_categories.py`: Utility function to extract the exact deduplicated category distributions within the Dev set.

### 4. Utilities, Debugging, Legacy and EDA

1. `src/main.py`: A centralized wrapper script designed to orchestrate the pipeline stages.

2. `src/verify_dev_score.py`: A sanity-check script used to quickly validate the structural integrity and F1 score of the generated `dev.txt` predictions against the local Dev set labels.

3. `src/debug_math.py`: A developmental utility to verify tensor shapes, attention mask broadcasting, and complex loss metric calculations when I was building the model architecture.

4. `tests/`: Contains unit tests.

4. `eda/`: Contains the code and plots for Exercise 2. Exploratory Data Analysis.