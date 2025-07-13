# Import necessary libraries --> Random Forest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA reduction
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train_pca, y_train)
y_pred_rf = rf.predict(X_test_pca)

# Performance Measures for Random Forest
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf, average='macro')
recall = recall_score(y_test, y_pred_rf, average='macro')
f1 = f1_score(y_test, y_pred_rf, average='macro')

print("Random Forest Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
from sklearn.metrics import classification_report

print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Visualizing Feature Importance
feature_importances = rf.feature_importances_
indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 5))
plt.title("Feature Importance")
plt.bar(range(X_train_pca.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X_train_pca.shape[1]), data.columns[:-1][indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

# Visualizing a Decision Tree from the Random Forest (optional)
# Let's visualize the first tree in the forest
from sklearn import tree
plt.figure(figsize=(20, 10))
tree.plot_tree(rf.estimators_[0], feature_names=data.columns[:-1], class_names=np.unique(y), filled=True)
plt.show()

# Define a function to predict the crop based on input features
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Create a dataframe for the input features
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })
    
    # Preprocess the input data
    input_data_scaled = scaler.transform(input_data)
    input_data_pca = pca.transform(input_data_scaled)
    
    # Predict the crop
    prediction = rf.predict(input_data_pca)
    return prediction[0]

# Example usage of the prediction function
predicted_crop = predict_crop(90, 42, 43, 20.87, 82.00, 6.50, 202.93)
print(f"The predicted crop using Random Forest is: {predicted_crop}")