# Import necessary libraries --> XGBoost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('../dataset/crop_recommendation.csv')

# Display the first few rows of the dataset
print(data.head())

# Preprocessing
X = data.drop('label', axis=1)  # Assuming 'label' is the target column
y = data['label']

# Encode the target variable y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train_encoded, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA reduction
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# XGBoost
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train_pca, y_train_encoded)
y_pred_xgb = xgb_model.predict(X_test_pca)

# Performance Measures for XGBoost
accuracy = accuracy_score(y_test, y_pred_xgb)
precision = precision_score(y_test, y_pred_xgb, average='macro')
recall = recall_score(y_test, y_pred_xgb, average='macro')
f1 = f1_score(y_test, y_pred_xgb, average='macro')

print("XGBoost Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Define a function to predict the crop based on input features
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Preprocess the input data
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })
    input_data_scaled = scaler.transform(input_data)
    input_data_pca = pca.transform(input_data_scaled)
    
    # Predict the crop
    prediction = xgb_model.predict(input_data_pca)
    return label_encoder.inverse_transform(prediction)[0]

# Example usage of the prediction function
predicted_crop = predict_crop(90, 42, 43, 20.87, 82.00, 6.50, 202.93)
print(f"The predicted crop using XGBoost is: {predicted_crop}")
