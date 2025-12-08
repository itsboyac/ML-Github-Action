import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle


def clean_and_preprocess(df):
    df = df.copy()
    
    # Drop unnecessary columns
    if 'CustomerId' in df.columns:
        df = df.drop(['CustomerId'], axis=1)
    if 'Surname' in df.columns:
        df = df.drop(['Surname'], axis=1)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Train bank customer churn prediction model')
    parser.add_argument('--data', type=str, default="kaggle-data/train.csv", help='Dataset filename')
    parser.add_argument('--n_estimators', type=int, default=75, help='Number of trees in Random Forest')
    parser.add_argument('--max_depth', type=int, default=None, help='Max tree depth (None for unlimited)')
    parser.add_argument('--test_size', type=float, default=0.3, help='Test set size (0.0-1.0)')
    parser.add_argument('--nrows', type=int, default=999, help='Number of rows to load from dataset')
    args = parser.parse_args()

    # Load Data
    bank_df = pd.read_csv(args.data, index_col="id", nrows=args.nrows)
    
    # Preprocess
    bank_df = clean_and_preprocess(bank_df)

    # Split Data
    # Target variable is 'Exited' (customer churn)
    X = bank_df.drop(['Exited'], axis=1)
    y = bank_df['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=125
    )

    # Identify categorical and numerical columns
    cat_cols = ['Geography', 'Gender']
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    # Impute missing values with mean
    imputer_num = SimpleImputer(strategy="mean")
    X_train_num = imputer_num.fit_transform(X_train[num_cols])
    X_test_num = imputer_num.transform(X_test[num_cols])
    
    # Scale to [0, 1] range
    scaler = MinMaxScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_test_num = scaler.transform(X_test_num)
    
    # Impute missing values with most frequent
    imputer_cat = SimpleImputer(strategy="most_frequent")
    X_train_cat = imputer_cat.fit_transform(X_train[cat_cols])
    X_test_cat = imputer_cat.transform(X_test[cat_cols])
    
    # Encode categorical variables
    encoders = {}
    X_train_cat_encoded = []
    X_test_cat_encoded = []
    
    for i, col in enumerate(cat_cols):
        le = LabelEncoder()
        X_train_cat_encoded.append(le.fit_transform(X_train_cat[:, i]))
        X_test_cat_encoded.append(le.transform(X_test_cat[:, i]))
        encoders[col] = le
    
    X_train_cat_encoded = np.column_stack(X_train_cat_encoded)
    X_test_cat_encoded = np.column_stack(X_test_cat_encoded)
    
    X_train_processed = np.hstack([X_train_num, X_train_cat_encoded])
    X_test_processed = np.hstack([X_test_num, X_test_cat_encoded])
    
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=125
    )
    model.fit(X_train_processed, y_train)
    
    predictions = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    print(f"Accuracy: {round(accuracy, 2) * 100}%")
    print(f"F1 Score: {round(f1, 2)}")
    print("="*50)

    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Bank Churn Prediction', fontsize=14, fontweight='bold')
    plt.savefig("model_results.png", dpi=120, bbox_inches='tight')
    print("Confusion matrix saved as 'model_results.png'")

    # Write metrics to file
    with open("metrics.txt", "w") as outfile:
        outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}\n\n")
    print("Metrics saved to 'metrics.txt'")

    # Save model and preprocessing objects
    model_artifacts = {
        'model': model,
        'imputer_num': imputer_num,
        'scaler': scaler,
        'imputer_cat': imputer_cat,
        'encoders': encoders,
        'num_cols': num_cols,
        'cat_cols': cat_cols
    }
    
    with open("bank_model.pkl", "wb") as f:
        pickle.dump(model_artifacts, f)
    print("Model and preprocessing objects saved as 'bank_model.pkl'")


if __name__ == "__main__":
    main()