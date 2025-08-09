# train_model.py
# This script reads our expanded and augmented dataset, processes the text,
# and uses GridSearchCV to find the best possible SVC model.

import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# --- Main function to wrap our logic for better error handling ---
def train_chatbot_model():
    """
    Reads, processes, and trains the chatbot model using GridSearchCV with an SVC.
    """
    print("--- Starting Model Training ---")

    # --- 1. Download necessary NLTK data ---
    print("Ensuring NLTK models are ready...")
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK models are ready.")


    # --- 2. Load and Preprocess the Data ---
    lemmatizer = WordNetLemmatizer()
    
    data_file_name = 'intents_expanded.json'
    print(f"Loading the expanded dataset: {data_file_name}...")
    with open(data_file_name, 'r', encoding='utf-8') as file:
        intents = json.load(file)
    print("Dataset loaded successfully.")

    training_sentences = []
    training_tags = []
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            lemmatized_words = [lemmatizer.lemmatize(w.lower()) for w in word_list]
            training_sentences.append(" ".join(lemmatized_words))
            training_tags.append(intent['tag'])

    print(f"\n{len(training_sentences)} total documents found.")
    print(f"{len(set(training_tags))} total classes found.")


    # --- 3. Build and Tune the Machine Learning Model ---
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SVC(class_weight='balanced', probability=True, random_state=42)),
    ])

    # Define a focused set of parameters to test.
    parameters = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [1, 10],
        'clf__kernel': ['linear'] # Linear kernel is often best for text data
    }

    grid_search = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1, verbose=2)

    X_train, X_test, y_train, y_test = train_test_split(
        training_sentences, training_tags, test_size=0.2, random_state=42, stratify=training_tags
    )

    print("\nStarting Hyperparameter Tuning with GridSearchCV on the expanded dataset...")
    grid_search.fit(X_train, y_train)

    print("\nBest parameters set found on development set:")
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel Accuracy on Test Data with Best Parameters: {accuracy * 100:.2f}%")
    
    # Print a detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


    # --- 4. Save the Best Trained Model ---
    # We will save the best model found by the grid search.
    with open('chatbot_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)

    print("\nBest model saved to chatbot_model.pkl")
    print("--- Model Training Complete ---")


# --- Run the training process and catch any errors ---
if __name__ == "__main__":
    try:
        train_chatbot_model()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
