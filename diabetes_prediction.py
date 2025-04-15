import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path=r"C:\Users\KRISH MALHOTRA\OneDrive\Documents\daibetes\diabetes.csv"):
    """
    Load and preprocess the diabetes dataset
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Replace 0 values with NaN for columns that shouldn't have 0
    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in zero_not_accepted:
        df[column] = df[column].replace(0, np.nan)
    
    # Fill missing values with median
    for column in df.columns:
        df[column] = df[column].fillna(df[column].median())
    
    return df

def prepare_data(df):
    """
    Prepare data for model training
    """
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_models(X_train, y_train):
    """
    Train multiple models and return them
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models and print results
    """
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get classification report
        class_report = classification_report(y_test, y_pred)
        
        # Get confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }
        
        print(f"\nResults for {name}:")
        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(class_report)
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()
    
    return results

def make_prediction(model, scaler, input_data):
    """
    Make prediction for new data
    """
    # Scale the input data
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)
    
    return prediction, probability

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    # Prepare data for training
    print("\nPreparing data for training...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Train models
    print("\nTraining models...")
    models = train_models(X_train, y_train)
    
    # Evaluate models
    print("\nEvaluating models...")
    results = evaluate_models(models, X_test, y_test)
    
    # Get user input for prediction
    print("\nEnter patient details for prediction:")
    try:
        pregnancies = float(input("Number of Pregnancies: "))
        glucose = float(input("Glucose Level: "))
        blood_pressure = float(input("Blood Pressure: "))
        skin_thickness = float(input("Skin Thickness: "))
        insulin = float(input("Insulin Level: "))
        bmi = float(input("BMI: "))
        pedigree = float(input("Diabetes Pedigree Function: "))
        age = float(input("Age: "))
        
        # Create input data array
        user_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]]
        
        # Make prediction using Random Forest model
        print("\nMaking prediction...")
        prediction, probability = make_prediction(models['Random Forest'], StandardScaler().fit(X_train), user_data)
        
        # Print results
        print("\nPrediction Results:")
        print(f"Diagnosis: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")
        print(f"Probability of being diabetic: {probability[0][1]:.2%}")
        print(f"Probability of not being diabetic: {probability[0][0]:.2%}")
        
    except ValueError:
        print("\nError: Please enter valid numerical values for all fields.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main() 