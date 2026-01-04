# Binary Classification Pipeline

This script performs an end-to-end machine learning workflow to predict a binary target variable. It merges two datasets, preprocesses the data, and evaluates three different classification models: Logistic Regression, Random Forest, and Support Vector Machine (SVM).

## Overview

The pipeline is designed to handle imbalanced datasets by using balanced class weights across all models. It employs a robust evaluation strategy that includes both Cross-Validation and a Grouped Hold-out split to ensure reliable performance metrics.

## Dependencies

The script requires Python 3 and the following libraries:

*   pandas
*   numpy
*   scikit-learn

You can install them via pip:

```bash
pip install pandas numpy scikit-learn
```

## Data Requirements

The script expects two CSV files located in a `../data/` directory relative to the script:

1.  `table_1.csv`
2.  `table_2.csv`

**File Structure:**
*   Both files must contain an `ID` column (used for merging).
*   The merged dataset must contain a target column named `Type`, with values "y" (positive) and "n" (negative).
*   The script automatically detects numerical and categorical columns for processing.

## Usage

Run the script from the command line:

```bash
python main.py
```

## Methodology

### 1. Data Preparation
*   **Merging:** Joins `table_1` and `table_2` on the `ID` column.
*   **Encoding:** Converts the "y/n" target into binary 1/0 integers.
*   **Preprocessing:**
    *   **Numeric features:** Imputed with the median and standardized (scaled).
    *   **Categorical features:** Imputed with the most frequent value and One-Hot encoded.

### 2. Models
The script trains and compares three models:
*   **Logistic Regression:** A linear baseline.
*   **Random Forest:** An ensemble method to capture non-linear relationships.
*   **SVM (RBF Kernel):** Support Vector Machine for complex decision boundaries.

All models are configured with `class_weight="balanced"` to penalize errors on the minority class more heavily.

### 3. Evaluation
The script uses two distinct evaluation methods:

*   **Cross-Validation:** Uses Stratified K-Fold (5 splits) to calculate average ROC-AUC, PR-AUC, Balanced Accuracy, and F1 scores.
*   **Hold-out Test:** Uses a `GroupShuffleSplit`. This splits the data into train and test sets while ensuring that records sharing the same `ID` do not leak between the training and testing sets. This provides a final "real-world" performance check.

## Output

The script prints the following to the console:
1.  Average metrics (ROC-AUC, PR-AUC, F1, Balanced Accuracy) with standard deviations for each model based on Cross-Validation.
2.  Detailed classification reports and confusion matrices for the final hold-out test set.
