import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import preprocess_data

# Function to apply SMOTE
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

# Function to apply ADASYN (optional)
def apply_adasyn(X_train, y_train):
    adasyn = ADASYN(random_state=42)
    X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

# Function to train Naïve Bayes
def train_naive_bayes(X_train, y_train, alpha=1.0):
    nb_model = MultinomialNB(alpha=alpha)  # Using MultinomialNB for text classification
    nb_model.fit(X_train, y_train)
    return nb_model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, df, test_indices):
    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    precision = precision_score(y_test, y_pred, average='weighted')  # multi class weighting
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # plotting confusion matrix on heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Hate Speech", "Offensive Language", "Neutral"],
                yticklabels=["Hate Speech", "Offensive Language", "Neutral"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Bias & Ethical Analysis
    analyze_bias(y_test, y_pred, df, test_indices)  # Pass df and test_indices


# Bias & Ethical Analysis: Detect false positives and false negatives
def analyze_bias(y_test, y_pred, df, test_indices):
    """Detect and analyze false positives and false negatives for bias"""
    
    # Identify false positives (predicted Hate Speech, but it was not)
    false_positive_idx = (y_pred == 0) & (y_test != 0)
    false_negative_idx = (y_pred != 0) & (y_test == 0)

    # Use test_indices to extract the corresponding tweets in the test set
    false_positive_tweets = df.iloc[test_indices[false_positive_idx]]['tweet']
    false_negative_tweets = df.iloc[test_indices[false_negative_idx]]['tweet']

    # Print some example false positives and false negatives
    print("\nFalse Positive Examples (Predicted Hate Speech but was not):")
    print(false_positive_tweets.head())  # Print a few examples
    
    print("\nFalse Negative Examples (Predicted not Hate Speech but was):")
    print(false_negative_tweets.head())  # Print a few examples

    # Fairness Metrics (Recall for each class)
    print("\nFairness Metrics (Precision, Recall, F1-Score for each class):")
    print(f"Recall for Hate Speech: {recall_score(y_test, y_pred, pos_label=0, average='weighted'):.4f}")
    print(f"Recall for Offensive Language: {recall_score(y_test, y_pred, pos_label=1, average='weighted'):.4f}")
    print(f"Recall for Neutral: {recall_score(y_test, y_pred, pos_label=2, average='weighted'):.4f}")
    
    # We can also use classification report for more details
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Hate Speech", "Offensive Language", "Neutral"]))

# Fine-tuning the model: Hyperparameter Tuning for Naïve Bayes (Alpha)
def hyperparameter_tuning(X_train_res, y_train_res, X_test, y_test):
    alpha_values = [0.1, 0.5, 1.0, 2.0]
    best_alpha = 1.0
    best_f1_score = 0.0

    for alpha in alpha_values:
        nb_model = train_naive_bayes(X_train_res, y_train_res, alpha)
        y_pred = nb_model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        if f1 > best_f1_score:
            best_f1_score = f1
            best_alpha = alpha

    print(f"Best alpha value: {best_alpha}, with F1 Score: {best_f1_score:.4f}")
    return best_alpha

# Adjusting the threshold for multiclass classification
def adjust_threshold(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)

    for i, label in enumerate(["Hate Speech", "Offensive Language", "Neutral"]):
        precision, recall, thresholds = precision_recall_curve(y_test == i, y_prob[:, i])
        plt.plot(thresholds, precision[:-1], label=f'Precision - {label}')
        plt.plot(thresholds, recall[:-1], label=f'Recall - {label}')

    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall Curve for Each Class')
    plt.legend()
    plt.show()

    new_thresholds = [0.6, 0.6, 0.6]
    y_pred_adjusted = []

    for i in range(len(y_test)):
        prob_class = y_prob[i, :]
        pred_class = (prob_class > new_thresholds).astype(int)
        y_pred_adjusted.append(np.argmax(pred_class))

    return np.array(y_pred_adjusted)

# Data Visualization: Class Distribution
def plot_class_distribution(y):
    class_counts = pd.Series(y).value_counts()
    class_labels = ["Hate Speech", "Offensive Language", "Neutral"]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_labels, y=class_counts.values, palette='viridis')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.show()

# Data Visualization: Precision-Recall Curve
def plot_precision_recall_curve(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)
    plt.figure(figsize=(8, 6))

    for i, label in enumerate(["Hate Speech", "Offensive Language", "Neutral"]):
        precision, recall, thresholds = precision_recall_curve(y_test == i, y_prob[:, i])
        plt.plot(thresholds, precision[:-1], label=f'Precision - {label}')
        plt.plot(thresholds, recall[:-1], label=f'Recall - {label}')

    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall Curve for Each Class')
    plt.legend()
    plt.show()

# Data Visualization: Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# BERT feature extraction
def extract_bert_features(texts, tokenizer, model, device):
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()  # Convert numpy array to list of strings
    elif isinstance(texts, torch.sparse.Tensor):
        texts = texts.to_dense().tolist()  # Convert sparse tensor to dense list
    elif isinstance(texts, pd.Series):
        texts = texts.tolist()  # If it's a pandas Series, convert to list

    if isinstance(texts, list) and all(isinstance(t, str) for t in texts):
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
    else:
        raise ValueError("Input must be a list of strings or a list of lists of strings.")

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

# Main script
if __name__ == "__main__":
    file_path = input("Please enter the path to the dataset CSV file: ")  # Adjust with your file path
    X_train, X_test, y_train, y_test, vectorizer, df, train_indices, test_indices = preprocess_data(file_path)

    print("Preprocessing completed!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # 2. apply SMOTE to handle class imbalance
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # 3. train Naïve Bayes model with resampled data
    nb_model = train_naive_bayes(X_train_res, y_train_res)

    # 4. hyperparameter tuning for Naïve Bayes (optional)
    best_alpha = hyperparameter_tuning(X_train_res, y_train_res, X_test, y_test)

    # 5. adjust the threshold to improve recall for the minority class
    y_pred_adjusted = adjust_threshold(nb_model, X_test, y_test)

    # 6. evaluate the model
    evaluate_model(nb_model, X_test, y_test, df, test_indices)  # Pass df and test indices

    # 7. visualizations
    plot_class_distribution(y_train)  # Class distribution
    plot_precision_recall_curve(nb_model, X_test, y_test)  # Precision-recall curve
    plot_confusion_matrix(y_test, y_pred_adjusted, ["Hate Speech", "Offensive Language", "Neutral"])  # Confusion matrix
