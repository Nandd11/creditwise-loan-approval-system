import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_path, target_column):
    print("⏳ Loading data and executing preprocessing pipeline...")
    
    # 1. Load the raw loan dataset
    df = pd.read_csv(data_path)
    
    # 2. Handle missing entries (Imputation)
    # Drop rows where the target prediction column is blank
    df = df.dropna(subset=[target_column])
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # For features, separate numeric and text data to fill blanks safely
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # Fill missing numeric values with the column median
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
        
    # Fill missing text values with the most frequent entry (mode)
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])
    
    # 3. Handle categorical columns automatically (One-Hot Encoding)
    X = pd.get_dummies(X, drop_first=True)
    
    # 4. Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Scale numeric values uniformly so features are balanced
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back into readable pandas DataFrames
    X_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_final = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_final, X_test_final, y_train, y_test, scaler