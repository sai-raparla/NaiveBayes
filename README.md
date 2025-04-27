# Naive Bayes - Final Project

## Mid Semester Progress Slides: [Naive Bayes](ProgressSlides.pdf)

## Data Set: [Kaggle Hate Speech](https://www.kaggle.com/datasets/yashdogra/toxic-tweets/data)


### Overview

This project implements an AI-powered content moderation system that classifies text into three categories:
1. **Hate Speech**
2. **Offensive Language**
3. **Neutral Content**

The system uses **Naïve Bayes** for classification and includes various preprocessing steps, such as removing stopwords, cleaning text, and vectorizing the text using **TF-IDF**. Additionally, the model handles class imbalance with **SMOTE** and evaluates its performance using standard metrics like **accuracy**, **precision**, **recall**, and **F1-score**.

### Requirements

To run this code, you will need the following libraries:

- **Python** (preferably Python 3.x)
- **pandas**: for data manipulation
- **nltk**: for natural language processing
- **sklearn**: for machine learning algorithms and evaluation
- **imblearn**: for handling imbalanced classes (SMOTE)
- **matplotlib** and **seaborn**: for visualization

To install these dependencies, you can use the following:

```bash
pip install pandas nltk scikit-learn imbalanced-learn matplotlib seaborn


### File Structure

- **preprocessing.py**: Contains functions to preprocess the data (text cleaning, stopword removal, text vectorization).
- **model_training.py**: Contains functions to train and evaluate the Naïve Bayes model, perform SMOTE, adjust thresholds, and generate visualizations.


### Running the Code

#### Step 1: Preprocess the Data

The script starts by loading the dataset, cleaning the text, and vectorizing it into numerical features using **TF-IDF**.

**Change the `file_path` in the `preprocessing.py` script**:

By default, the `file_path` is set to `'C:/Users/princ/NaiveBayes/labeled_data.csv'`, which may not work on another system. To make it flexible, you should either:

- **Update the `file_path`** with the correct path to your dataset, or
- **Modify the script to use a relative file path** or prompt the user for the file location.

#### Flexible File Path Solution:

Instead of hardcoding the `file_path`, you can modify the code to take the file path as an argument. This will make it easier for other users to run the code without modifying it.

**Change the following in `preprocessing.py`:**

```python
import os

# asks for the file path
file_path = input("Please enter the path to the dataset CSV file: ")

# Alternatively, use a relative path if the dataset is in the same directory as the script
# file_path = os.path.join(os.getcwd(), 'labeled_data.csv')  # Uncomment if dataset is in the same directory


#### Step 2: Train the Model

The model uses **Naïve Bayes** to classify the text. You can choose to apply **SMOTE** to balance the class distribution.

#### Step 3: Evaluate the Model

After training the model, the script evaluates its performance using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**
- **Precision-Recall Curve**

These metrics will help you assess how well the model performs across different classes.

#### Step 4: Visualizations

The following visualizations are generated:
- **Class Distribution**: A bar chart showing how the dataset is distributed across the three classes.
- **Precision-Recall Curves**: A plot showing the trade-off between precision and recall for each class.
- **Confusion Matrix**: A heatmap showing the misclassifications between classes.

#### Step 5: Adjust the Model

The script also includes functionality to adjust the model’s **thresholds** to improve recall for the minority class (e.g., **Hate Speech**), which might be underrepresented in the data.


### Customization

You can customize the following parts of the code:

- **Custom Stopwords**: You can add additional stopwords relevant to your project in the `get_custom_stopwords()` function.
- **Threshold Adjustments**: The threshold for classifying **Hate Speech** or **Offensive Language** can be adjusted to improve recall for the minority class.

### Running Code Command

To run the model, simply execute the script from the command line:

```bash
python model_training.py