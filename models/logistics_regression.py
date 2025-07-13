# Import necessary libraries --> Logistic Regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_pca, y_train)
y_pred_lr = lr.predict(X_test_pca)

# Performance Measures for Logistic Regression
accuracy = accuracy_score(y_test, y_pred_lr)
precision = precision_score(y_test, y_pred_lr, average='macro')
recall = recall_score(y_test, y_pred_lr, average='macro')
f1 = f1_score(y_test, y_pred_lr, average='macro')

print("Logistic Regression Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
from sklearn.metrics import classification_report

print("Classification Report:")
print(classification_report(y_test, y_pred_lr))

# ROC Curve and AUC (if applicable, usually for binary classification)
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Binarize the output labels for multiclass ROC
y_test_binarized = label_binarize(y_test, classes=np.unique(y))
y_pred_proba = lr.predict_proba(X_test_pca)
n_classes = y_test_binarized.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotting the ROC Curve for each class
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-class')
plt.legend(loc='lower right')
plt.show()

# Create new data for prediction from the existing dataset
new_data = X_test[:5]  # Taking the first 5 samples from the test set as new data
print("New data for prediction (first 5 samples of the test set):")
print(new_data)

# Standardization (already done for X_test)
new_data_scaled = scaler.transform(new_data)

# PCA transformation (already done for X_test)
new_data_pca = pca.transform(new_data_scaled)

# Predictions on new data
predictions = lr.predict(new_data_pca)

print("Predictions on new data:")
print(predictions)

# Outputting the actual labels for comparison
actual_labels = y_test[:5]
print("Actual labels of new data:")
print(actual_labels)

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
    prediction = lr.predict(input_data_pca)
    return prediction[0]

# Example usage of the prediction function
predicted_crop = predict_crop(90, 42, 43, 20.87, 82.00, 6.50, 202.93)
print(f"The predicted crop is: {predicted_crop}")
