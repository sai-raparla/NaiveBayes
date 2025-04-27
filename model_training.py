import numpy as np  # Import NumPy
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import preprocess_data  # Import preprocess_data function from preprocessing.py

# Function to apply SMOTE
def apply_smote(X_train, y_train):
    """Apply SMOTE to oversample the minority class (Hate Speech - Class 0)"""
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

# Function to apply ADASYN (optional)
def apply_adasyn(X_train, y_train):
    """Apply ADASYN to handle class imbalance"""
    adasyn = ADASYN(random_state=42)
    X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

# Function to train Naïve Bayes
def train_naive_bayes(X_train, y_train, alpha=1.0):
    """Train a Naïve Bayes model"""
    nb_model = MultinomialNB(alpha=alpha)  # Using MultinomialNB for text classification
    nb_model.fit(X_train, y_train)
    return nb_model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    precision = precision_score(y_test, y_pred, average='weighted')  # Weighted for multiclass
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Hate Speech", "Offensive Language", "Neutral"],
                yticklabels=["Hate Speech", "Offensive Language", "Neutral"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Bias & Ethical Analysis
    analyze_bias(y_test, y_pred)

# Bias & Ethical Analysis: Detect false positives and false negatives
def analyze_bias(y_test, y_pred):
    """Detect and analyze false positives and false negatives for bias"""
    
    # Identify false positives (predicted Hate Speech, but it was not)
    false_positive_idx = (y_pred == 0) & (y_test != 0)
    false_negative_idx = (y_pred != 0) & (y_test == 0)
    
    # Print some example false positives and false negatives
    print("\nFalse Positive Examples (Predicted Hate Speech but was not):")
    print(pd.Series([x for i, x in enumerate(y_test) if false_positive_idx[i]]))
    
    print("\nFalse Negative Examples (Predicted not Hate Speech but was):")
    print(pd.Series([x for i, x in enumerate(y_test) if false_negative_idx[i]]))

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
    """Experiment with different alpha values for Naïve Bayes"""
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
    """Adjust the decision threshold for each class to improve recall for the minority class"""
    # Get the probability predictions for each class
    y_prob = model.predict_proba(X_test)
    
    # Plot precision-recall curve for each class
    for i, label in enumerate(["Hate Speech", "Offensive Language", "Neutral"]):
        precision, recall, thresholds = precision_recall_curve(y_test == i, y_prob[:, i])  # binary for each class
        
        # Plot the precision-recall curve for each class
        plt.plot(thresholds, precision[:-1], label=f'Precision - {label}')
        plt.plot(thresholds, recall[:-1], label=f'Recall - {label}')
        
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall Curve for Each Class')
    plt.legend()
    plt.show()

    # Choose a threshold based on the precision-recall trade-off
    new_thresholds = [0.6, 0.6, 0.6]  # Example thresholds for each class
    y_pred_adjusted = []
    
    for i in range(len(y_test)):
        prob_class = y_prob[i, :]
        pred_class = (prob_class > new_thresholds).astype(int)
        # Select the class with the highest probability
        y_pred_adjusted.append(np.argmax(pred_class))  # Convert back to class label
        
    return np.array(y_pred_adjusted)

# Data Visualization: Class Distribution
def plot_class_distribution(y):
    """Visualize the class distribution"""
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
    """Plot precision-recall curves for each class"""
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
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Main script
if __name__ == "__main__":
    # File path to your preprocessed CSV file (update the path as needed)
    file_path = input("Please enter the path to the dataset CSV file: ")  # Adjust with your file path

    # 1. Preprocess the data (imported from preprocessing.py)
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(file_path)

    # 2. Apply SMOTE to handle class imbalance
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # 3. Train Naïve Bayes model with resampled data
    nb_model = train_naive_bayes(X_train_res, y_train_res)

    # 4. Hyperparameter tuning for Naïve Bayes (optional)
    best_alpha = hyperparameter_tuning(X_train_res, y_train_res, X_test, y_test)

    # 5. Adjust the threshold to improve recall for minority class
    y_pred_adjusted = adjust_threshold(nb_model, X_test, y_test)

    # 6. Evaluate the model
    evaluate_model(nb_model, X_test, y_test)

    # 7. Visualizations
    plot_class_distribution(y_train)  # Class distribution
    plot_precision_recall_curve(nb_model, X_test, y_test)  # Precision-recall curve
    plot_confusion_matrix(y_test, y_pred_adjusted, ["Hate Speech", "Offensive Language", "Neutral"])  # Confusion matrix
