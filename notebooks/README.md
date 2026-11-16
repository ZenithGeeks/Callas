# Scam classifier notebook

`scam_classifier_phayathaibert.ipynb` cleans the scam call dataset and trains two models on it:

- the gate: is this call a scam or not
- the type: which scam type it is (only for scam calls)

Each model is trained two ways: a simple baseline (TF-IDF + logistic regression) and a fine-tuned PhayaThaiBERT (`clicknext/phayathaibert`). The notebook scores both on the same test set so you can compare them.

## Setup

You need Python 3.10 or newer and a GPU for part B. Part A runs on CPU.

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[ml]"
```

Install torch first. The normal Windows wheel has no GPU support.

## How to run

1. Put the data at `dataset/scam_dataset.csv`. If you only have an Excel file, save it as `dataset/scam_dataset.xlsx` and the notebook makes the csv for you.
2. Open the notebook, pick the `.venv` kernel, and run all cells from top to bottom.
3. A full run takes about 15 minutes on an RTX 5070 Ti. Most of that is cross-validation (step 12). Set `RUN_BERT_CV = False` in step 12 to skip the slow part.

## Files

| File | What it is |
|------|------------|
| `dataset/scam_dataset.csv` | Input. One call per row: `dialogue`, `scam_class`, `scam_type`. |
| `dataset/scam_dataset_generated.csv` | Optional input. Simulated calls from `data_pipeline`. See "Generated data". |
| `data/processed/dataset.csv` | All cleaned samples. |
| `data/processed/train.csv`, `valid.csv`, `test.csv` | The fixed 70 / 15 / 15 split. |
| `models/phayathaibert-scam-gate/` | Saved gate model. Not in git. |
| `models/phayathaibert-scam-type/` | Saved type model. Not in git. |
| `models/scam_classifier_metrics.json` | All scores from the run. Not in git. |

All csv files are UTF-8 with BOM, so Thai text looks right in Excel. The columns `speaker_turns` and `entities` hold JSON text because a csv cell cannot hold a list.

Each model folder also has `preprocessing.json`. It lists the text steps the model was trained with: the row text, the speaker letters, the masked entity labels, the space token and the length limit. `modules/nlp_core/phayathaibert.py` reads it to write the live call the same way, and refuses a model saved by an older notebook. With `NLP_CLASSIFIER=phayathaibert` the web demo uses the saved models; the `/live` page then shows a test-split call to read aloud.

## Steps

Part A prepares the data. Part B trains and scores.

| Step | What it does |
|------|--------------|
| Input format | One cell with everything the notebook assumes about the raw file. |
| 1. Data loading | Reads the csv and stops with a clear message if the format is wrong. |
| 2. EDA: data leakage | Counts things in the text that give away the label. |
| 3. Cleaning | Removes notes like `[วางสาย]` and narration, normalizes Thai text with PyThaiNLP, rewrites each call as `A: ...` / `B: ...`. |
| 4. Label mapping | Maps the dataset's type names to the 5 spec types. Drops duplicates. |
| 5. Near-duplicate detection | Groups calls that are almost the same text. |
| 6. Split | 70 / 15 / 15 with seed 42. A group never crosses two sets. |
| 7. Export | Writes the csv files. |
| 8. Baseline | TF-IDF + logistic regression. Also checks which speaker the gate relies on, and whether generated rows in train help. |
| 9. Tokenization | Loads the PhayaThaiBERT tokenizer and checks text length. |
| 10. Gate model | Fine-tunes PhayaThaiBERT for scam / not scam. |
| 11. Type model | Fine-tunes PhayaThaiBERT for the scam type. |
| 12. Cross-validation | Trains and scores 5 times on different folds. |
| 13. Evaluation | Final scores on the test set. |
| 14. Inference | Loads the saved models and tries two example calls. |

Train, valid and test have different jobs. The model learns from train. Valid picks the best epoch. Test is used once, for the final score.

## Settings you may want to change

| Setting | Where | What it does |
|---------|-------|--------------|
| `DATA_FILE`, `COLUMNS`, `ROLE_TAGS` | Input format cell | Change these when the file name, column names or speaker tags change. |
| `TYPE_MAP` | Step 4 | Which dataset type goes to which spec type. |
| `MIN_TYPE_SAMPLES`, `MIN_TYPE_EVAL_SAMPLES` | First cell | How many samples a type needs before the type model trains on it. |
| `REBUILD_SPLIT` | First cell | `True` makes a new split. `False` keeps the files on disk. |
| `KEEP_ORIGINAL_TEST` | First cell | With generated data, `True` keeps the hand-written test set. See "Generated data". |
| `GENERATED_FILE` | Input format cell | The generated csv. Set it to `None` to train on hand-written data only. |
| `RUN_BERT_CV`, `N_FOLDS` | Step 12 | Turn PhayaThaiBERT cross-validation on or off. |
| `GATE_MIN_CONFIDENCE`, `TYPE_MIN_CONFIDENCE` | Step 14 and start of part B | Below these the model answers "unsure". |

## Generated data

`python -m data_pipeline.synthesize --export-notebook dataset/scam_dataset_generated.csv` writes simulated calls in the same format (see [`modules/data_pipeline/README.md`](../modules/data_pipeline/README.md#export-for-the-classifier-notebook)). The notebook reads that file when it exists. Without it, the notebook gives back `data/processed` unchanged.

- Rows with the same `group_hint` (call type and scenario) always land in one split.
- With `KEEP_ORIGINAL_TEST = True`, every hand-written row keeps the split it has in `data/processed/{train,valid,test}.csv`.
- A generated row follows the hand-written rows of its group. It is dropped if that split is test, or if its group has hand-written rows in two splits (the notebook prints a warning). Groups with no hand-written row go to train and valid only.

## Results of the last run

Run on 2026-10-13 with the demo dataset: 377 calls in the file, 372 after cleaning. Split: 260 train, 57 valid, 55 test.

The type model was trained on `fake_authority` and `phishing_otp` only. `romance`, `investment` and `lottery` do not have enough samples yet.

Macro-F1 (higher is better, 1.0 is perfect):

| Task | Model | Test set | 95% range on test | 5-fold mean | 5-fold lowest and highest |
|------|-------|----------|-------------------|-------------|---------------------------|
| Gate | Baseline | 0.976 | 0.91 to 1.00 | 0.983 | 0.978 to 1.000 |
| Gate | PhayaThaiBERT | 0.952 | 0.87 to 1.00 | 0.918 | 0.726 to 1.000 |
| Type | Baseline | 0.862 | 0.64 to 1.00 | 0.602 | 0.495 to 0.689 |
| Type | PhayaThaiBERT | 0.862 | 0.68 to 1.00 | 0.850 | 0.762 to 0.913 |

What this says:

- Gate: the baseline is as good as PhayaThaiBERT or better, and it is steadier across folds. Fine-tuning did not help here.
- Type: on the test set both models get the same score. In cross-validation PhayaThaiBERT is clearly better (0.850 against 0.602). The test set has only 8 `phishing_otp` calls, so trust the cross-validation number more.
- The gate baseline scores 0.976 on valid when it reads only the receiver's turns, the same as with the full call. So the gate can be solved from how the receiver answers.
- The type model gave a type to all 39 scam calls that belong to none of its types. It cannot say "other" yet.

## Known limits

- The data is generated and small. These are demo numbers.
- In almost every scam call the receiver sees the scam and refuses. Real victims often go along. New data needs scam calls where the receiver does what the caller says, and normal calls where the receiver is careful.
- Many calls come from the same template with a few words changed. The split keeps near copies together, but looser copies can still be in both train and test.
- Some calls with the same text had different types in the dataset. Step 4 prints and drops them.
- The cleaning rules for notes and narration were written for this demo file. Check step 2 again when the data changes.

## When the data changes

- More rows, same format: replace the csv, keep `REBUILD_SPLIT = True`, run all cells. Add new type names to `TYPE_MAP`.
- New format: edit the Input format cell. If speakers are marked in a new way, rewrite `parse_dialogue` in that cell.
- A new split changes which calls are in test. Old scores and new scores cannot be compared, so run every model again.
