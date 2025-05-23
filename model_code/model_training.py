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
from preprocessing import preprocess_data

# SMOTE
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

# Function to train Naïve Bayes
def train_naive_bayes(X_train, y_train, alpha=1.0):
    nb_model = MultinomialNB(alpha=alpha)  # Using MultinomialNB for text classification
    nb_model.fit(X_train, y_train)
    return nb_model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    # Predict on the test data
    y_pred = model.predict(X_test)

    # Accuracy - out of all predicitions how many were true
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Precision - out of all positive predictions, how many were true
    precision = precision_score(y_test, y_pred, average='weighted')  # multi class weighting

    # Recall 
    recall = recall_score(y_test, y_pred, average='weighted')

    # F1 Score
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

    return accuracy, precision, recall, f1

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


if __name__ == "__main__":
    file_path = '../labeled_data.csv'
    X_train, X_test, y_train, y_test = preprocess_data(file_path)

    print("Preprocessing completed!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # SMOTE for data inbalance
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    print("finished applying smote")

    # naive bayes training
    nb_model = train_naive_bayes(X_train_res, y_train_res)

    print('finished training')

    # 5. adjust the threshold to improve recall for the minority class
    y_pred_adjusted = adjust_threshold(nb_model, X_test, y_test)

    print('finished adjusting thresholds')

    # 6. evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(nb_model, X_test, y_test) 

    print("finished evaluating")

    # 7. visualizations
    plot_class_distribution(y_train)  # Class distribution
    plot_precision_recall_curve(nb_model, X_test, y_test)  # Precision-recall curve
    plot_confusion_matrix(y_test, y_pred_adjusted, ["Hate Speech", "Offensive Language", "Neutral"])  # Confusion matrix
