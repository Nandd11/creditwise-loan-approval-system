import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score
from preprocess import load_and_preprocess_data

def run_training():
    # Define exact paths and settings
    data_path = "data/loan_approval_data.csv"   # We will verify this name next!
    target_col = "Loan_Approved"       # The common target name for credit files
    
    # 1. Fetch clean data splits using our preprocess script
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path, target_col)
    
    print("🌲 Initializing and training Random Forest Classifier...")
    # 2. Train a robust Classifier
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    
    # 3. Generate predictions and evaluate performance
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    
    print("\n🚀 Training Complete!")
    print(f"📊 Overall Accuracy Achieved: {accuracy * 100:.1f}%")
    print("\n📋 Full Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 4. Save the production artifacts so we can use them later
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/loan_model.pkl")
    joblib.dump(scaler, "models/loan_scaler.pkl")
    print("💾 Model and Scaler artifacts successfully saved to /models directory!")

if __name__ == "__main__":
    run_training()