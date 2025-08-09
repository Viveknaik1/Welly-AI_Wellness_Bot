# train_model.py
# This script trains the intent classification model for the chatbot.
# It uses a Support Vector Machine (SVC) with GridSearchCV for hyperparameter tuning.

import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def train_chatbot_model():
    """
    Main function to read the processed dataset, train the model, and save it.
    """
    print("--- Starting Model Training ---")

    # Ensure NLTK data is available for tokenization and lemmatization
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    # Load the final, expanded dataset
    with open('intents_expanded.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)

    # Preprocess the text data: tokenize, lemmatize, and format for training
    lemmatizer = WordNetLemmatizer()
    training_sentences = []
    training_tags = []
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            lemmatized_words = [lemmatizer.lemmatize(w.lower()) for w in word_list]
            training_sentences.append(" ".join(lemmatized_words))
            training_tags.append(intent['tag'])

    print(f"\nTraining on {len(training_sentences)} total examples.")

    # Define the model pipeline: text vectorization followed by classification
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SVC(class_weight='balanced', probability=True, random_state=42)),
    ])

    # Set up parameters for GridSearchCV to find the best model configuration
    parameters = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [1, 10],
        'clf__kernel': ['linear']
    }

    grid_search = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1, verbose=2)

    # Split data into training and testing sets to evaluate performance
    X_train, X_test, y_train, y_test = train_test_split(
        training_sentences, training_tags, test_size=0.2, random_state=42, stratify=training_tags
    )

    print("\nStarting Hyperparameter Tuning...")
    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:")
    print(grid_search.best_params_)

    # Evaluate the best model on the unseen test data
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel Accuracy on Test Data: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Save the final trained model for use in the chatbot application
    with open('chatbot_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)

    print("\nBest model saved to chatbot_model.pkl")
    print("--- Model Training Complete ---")

# Training function is called only when the script is executed directly.
if __name__ == "__main__":
    try:
        train_chatbot_model()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
