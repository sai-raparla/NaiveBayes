import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

# Load the dataset
def load_data(file_path):
    """Loads dataset from a CSV file"""
    df = pd.read_csv(file_path)
    print(df.head())  # To check the first few rows of the dataset
    return df

# 1. Custom Stopwords List
def get_custom_stopwords():
    """Returns the default stopwords with custom additions"""
    # Get the default stopwords list
    stop_words = set(stopwords.words('english'))

    # Add custom stopwords relevant to the project
    custom_stopwords = [
        "just", "really", "like", "know", "dont", "didnt", "want", "think", "actually",
        "i'm", "i've", "can", "could", "youre", "thats", "gonna", "would", "hasnt", "wasnt",
        "lol", "lmao", "rofl", "tbh", "fyi", "brb", "omg", "btw", "idk", "lmk", "smh", "bff",
        "yolo", "u", "ur", "r", "i", "you", "he", "she", "it", "they", "we", "our", "ours", "mine", "his", 
        "hers", "their", "theirs", "a", "an", "the", "this", "that", "these", "those", "all", "some", "any",
        "please", "thank", "sorry", "hey", "hello", "good", "bad", "okay", "fine", "sure", "great", "love", "hate",
        "is", "are", "was", "were", "be", "been", "being", "have", "had", "has", "having", "do", "does", "did", "doing",
        "not", "never", "none", "no", "nothing", "nobody", "nothin", "neither", "123", "456", "789", ".", "!", "?", 
        "bro", "sis", "fam", "dude", "homie", "girl", "guy", "man", "woman", "people", "kids", "guys", "ladies"
    ]
    
    # Combine the default stopwords with the custom ones
    stop_words.update(custom_stopwords)
    return stop_words

# 2. Text Cleaning
def clean_text(text, stop_words):
    """Cleans the input text"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions (@username) and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove non-alphanumeric characters (excluding spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# 3. Vectorization using TfidfVectorizer
def vectorize_text(text_data):
    """Converts text data into numerical features using TfidfVectorizer"""
    vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed
    X = vectorizer.fit_transform(text_data)  # Keep as sparse matrix
    return X, vectorizer

# 4. Preprocessing Pipeline
def preprocess_data(file_path):
    """Complete preprocessing pipeline"""
    # Load data
    df = load_data(file_path)
    
    # Use the correct column names ('tweet' for text and 'class' for labels)
    texts = df['tweet']  # Update this to 'tweet' for tweet text
    labels = df['class']  # Update this to 'class' for the label (the target)
    
    # Get custom stopwords
    stop_words = get_custom_stopwords()
    
    # Clean text data
    df['cleaned_text'] = texts.apply(lambda x: clean_text(x, stop_words))
    
    # Vectorize the cleaned text
    X, vectorizer = vectorize_text(df['cleaned_text'])
    
    # Prepare labels
    y = df['class'].values  # Using the 'class' column for labels
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)  # Encode the training labels
    y_test_encoded = le.transform(y_test)  # Encode the test labels
    
    return X_train, X_test, y_train_encoded, y_test_encoded, vectorizer

if __name__ == "__main__":
    file_path = 'C:/Users/princ/NaiveBayes/labeled_data.csv'  # Path to the uploaded file
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(file_path)
    
    print("Preprocessing completed!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
