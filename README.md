# Bank Churn Prediction

This project implements a complete Machine Learning pipeline to predict bank customer churn. It uses a Random Forest classifier to identify customers who are likely to exit the bank's services based on various demographic and financial features.

## Project Structure

The repository contains the following key files:

- **`train.py`**: The main script that executes the end-to-end ML pipeline.
- **`requirements.txt`**: List of Python dependencies required to run the project.
- **`kaggle-data/`**: Directory containing the dataset (`train.csv`).

## Setup and Installation
 **Install dependencies**:
    Using a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the machine learning pipeline, execute the `train.py` script:

```bash
python train.py
```

When successfully executed, the script will:
- Display the model's Accuracy and F1 Score in the console.
- Generate a confusion matrix plot saved as `model_results.png`.
- Save the evaluation metrics to text file `metrics.txt`.
- Save the trained pipeline as a binary file `bank_pipeline.skops`.

## Pipeline Explanation

The `train.py` script performs the following steps in detail:

### 1. Data Loading to Initial Processing
- **Source**: Reads data from `kaggle-data/train.csv`.
- **Filtering**: Drops irrelevant columns `CustomerId` and `Surname` that do not contribute to churn prediction.
- **Splitting**: Separates the data into features (`X`) and target variable (`y` = "Exited").
- **Train-Test Split**: Divides the data into training (70%) and testing (30%) sets to ensure the model is evaluated on unseen data.

### 2. Data Preprocessing
The pipeline handles different data types automatically:

- **Numerical Features** (e.g., CreditScore, Age, Balance):
    - **Imputation**: Missing values are filled with the *mean* of the column.
    - **Scaling**: Values are normalized using `MinMaxScaler` to bring them into a standard range.

- **Categorical Features** (e.g., Geography, Gender):
    - **Imputation**: Missing values are filled with the *most frequent* value.
    - **Encoding**: Converted to numerical format using `OrdinalEncoder`.

These preprocessing steps are wrapped in a `ColumnTransformer` to apply them correctly to the respective columns.

### 3. Feature Selection
- The pipeline uses `SelectKBest` with the Chi-Squared (`chi2`) statistical test to select the most relevant features for the model.

### 4. Model Training
- **Algorithm**: `RandomForestClassifier`.
- **Configuration**: Uses 75 estimators (decision trees) to build a robust ensemble model.
- **Pipeline Integration**: The preprocessing, feature selection, and model are bundled into a single `scikit-learn` Pipeline. This ensures that all transformations applied to training data are identically applied to testing data.

### 5. Evaluation and Artifacts
After training, the model is evaluated on the test set:
- **Metrics**: Calculates **Accuracy** (overall correctness) and **F1 Score** (balance between precision and recall).
- **Visualization**: Generates and saves a Confusion Matrix (`model_results.png`) to visualize true positives, false positives, etc.
- **Persistence**: 
    - Writes metrics to `metrics.txt`.
    - Saves the entire trained pipeline to `bank_pipeline.skops` using the `skops` library for secure persistence.
