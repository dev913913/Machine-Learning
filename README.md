# Machine Learning Lab Practice Environment

This repository provides a ready-to-run Python environment and a structured set of lab exercises based on the **ML Lab Syllabus**. Each exercise has its own script under `lab_exercises/` with brief explanations and runnable code.

## Quick start

1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run any exercise:
   ```bash
   python lab_exercises/exercise_01_csv_basics.py
   ```

## Exercises mapping

| # | Syllabus topic | Script |
|---|---|---|
| 1 | Read numeric CSV + basic operations | `lab_exercises/exercise_01_csv_basics.py` |
| 2 | Decision Tree classification | `lab_exercises/exercise_02_decision_tree.py` |
| 3 | Random Forest classification | `lab_exercises/exercise_03_random_forest.py` |
| 4 | Naive Bayes from CSV + accuracy | `lab_exercises/exercise_04_naive_bayes_csv.py` |
| 5 | Document classification with Naive Bayes | `lab_exercises/exercise_05_naive_bayes_documents.py` |
| 6 | Bayesian Network for medical data | `lab_exercises/exercise_06_bayesian_network.py` |
| 7 | k-NN on iris dataset | `lab_exercises/exercise_07_knn_iris.py` |
| 8 | K-means clustering | `lab_exercises/exercise_08_kmeans.py` |
| 9 | SVM classification | `lab_exercises/exercise_09_svm.py` |
| 10 | Hierarchical clustering | `lab_exercises/exercise_10_hierarchical_clustering.py` |

## Data files

Sample datasets live in `data/`. See `data/README.md` for a short description of each file.

## Notes

- Exercises are intentionally small and pedagogical so you can modify them for practice.
- The Bayesian network example uses `pgmpy` to define a simple medical model; you can replace the dataset with the standard heart disease dataset when you have it.
