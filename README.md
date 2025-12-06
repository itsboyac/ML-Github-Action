# Bank Churn Prediction with MLOps

This project implements a complete Machine Learning pipeline to predict bank customer churn. Beyond just the code, it uses **GitHub Actions** to automate the training and evaluation process, following **MLOps (Machine Learning Operations)** best practices.

## GitHub Actions & MLOps Explanation

This project is not just a script you run locally; it is an automated pipeline. Here is why we structured it this way:

### 1. Why create a YAML file? (`.github/workflows/main.yml`)
The `.yaml` file serves as the **blueprint** for automation. We create it to define a "Workflow" that GitHub's servers can understand. 
- **What it does**: It tells GitHub *when* to run (e.g., on every "push" or "pull request") and *what* to do (e.g., "install python", "run train.py").
- **Why it's needed**: Without this file, GitHub is just a place to store code. With it, GitHub becomes an automation server that can test and validate our work.

### 2. What does a GitHub Action do?
Think of a GitHub Action as an **automated robot** that watches your repository.
- When you make a change (like pushing new code), the Action "wakes up".
- It spins up a temporary computer (called a **runner**, usually `ubuntu-latest`).
- It follows the instructions in your YAML file step-by-step to check your code.

### 3. Why is this useful here? (Continuous Machine Learning)
We are using a methodology called **CML (Continuous Machine Learning)**. This is incredibly useful for three reasons:

*   **Automation**: We don't have to manually run `train.py` on our laptops every time we change a feature. The system does it for us.
*   **Consistency**: The code runs in a clean, isolated environment (the runner). If it works there, we know it's not just "working on my machine."
*   **Immediate Feedback**: The most powerful feature is the **Report**. 
    - When you open a Pull Request, the Action trains the model and *comments back on your PR* with the results.
    - It generates a report containing the **Accuracy**, **F1 Score**, and a **Confusion Matrix** plot.
    - This allows us to see if a code change improved or worsened the model *before* we merge it.

### 4. What is `requirements.txt`?
Since the GitHub Action spins up a fresh, blank computer (runner) every time, it doesn't have Python libraries like `pandas` or `scikit-learn` installed by default.
- The `requirements.txt` is the **shopping list** for the runner.
- The step `pip install -r requirements.txt` tells the runner to download exactly the same libraries we use locally, ensuring the model runs exactly the same way in the cloud.

---

## Project Structure

- **`.github/workflows/main.yml`**: The CI/CD configuration that defines the MLOps pipeline.
- **`train.py`**: The script that loads data, cleans it, trains the Random Forest model, and saves metrics.
- **`requirements.txt`**: List of dependencies for the environment.
- **`kaggle-data/`**: Directory containing the `train.csv` dataset.

## How the Pipeline Works (`train.py`)

When the GitHub Action triggers these steps, `train.py` executes logic:
1.  **Data Loading**: Reads `train.csv` and removes irrelevant ID columns.
2.  **Preprocessing**:
    - Fills missing numbers with the *mean*.
    - Scales numerical features (0-1).
    - Fills missing categories with *most frequent* and encodes them.
3.  **Feature Selection**: Uses Chi-Squared test to pick the best features.
4.  **Training**: Trains a **Random Forest Classifier**.
5.  **Evaluation**: Calculates accuracy and saves a plot (`model_results.png`) and text metrics (`metrics.txt`) which the CML tool picks up to create the report.
