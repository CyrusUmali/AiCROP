import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, r2_score
import pickle
import os
from pathlib import Path

# Load dataset
dataset_path = 'dataset/crop_recommendation.csv'
df = pd.read_csv(dataset_path)

# Encode the categorical label into numerical values
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Split data into features and target
X = df.drop('label', axis=1)
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize StandardScaler and fit_transform on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data (don't fit again, only transform)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

metrics = {}

for name, model in models.items():
    if name == 'Logistic Regression':
        # Train Logistic Regression with scaled data
        model.fit(X_train_scaled, y_train)
        X_test_used = X_test_scaled  # Use scaled test data for Logistic Regression
    else:
        # Train Decision Tree and Random Forest with original data
        model.fit(X_train, y_train)
        X_test_used = X_test  # Use original test data for Decision Tree and Random Forest

    # Predict using the trained model
    y_pred = model.predict(X_test_used)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    r2 = r2_score(y_test, y_pred)

    metrics[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'r2_score': r2
    }

    # Save Decision Tree model and visualize up to depth 3
    if name == 'Decision Tree':
        dot_file_path = 'precomputation/decision_tree.dot'
        os.makedirs(os.path.dirname(dot_file_path), exist_ok=True)
        export_graphviz(model, out_file=dot_file_path, 
                       feature_names=X.columns, 
                       class_names=le.classes_, 
                       filled=True, 
                       rounded=True, 
                       max_depth=3)

# Create precomputation directory if it doesn't exist
os.makedirs('precomputation', exist_ok=True)

# Save all artifacts
artifacts = {
    'models': models,
    'metrics': metrics,
    'label_encoder': le,
    'scaler': scaler,
    'feature_columns': X.columns.tolist()
}

with open('precomputation/training_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("Training complete. All artifacts saved to precomputation/training_artifacts.pkl")
print("\nModel Metrics:")
for model_name, metric in metrics.items():
    print(f"\n{model_name}:")
    print(f"  Accuracy: {metric['accuracy']:.4f}")
    print(f"  Precision: {metric['precision']:.4f}")
    print(f"  R2 Score: {metric['r2_score']:.4f}")