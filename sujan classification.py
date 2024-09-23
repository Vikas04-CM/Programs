import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load the Wine Quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Select features and target variable
X = data.drop('quality', axis=1)  # Features
y = (data['quality'] > 5).astype(int)  # Target variable (1 for good quality, 0 for bad quality)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and fit the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Bad Quality', 'Good Quality']))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Bad Quality', 'Good Quality'], yticklabels=['Bad Quality', 'Good Quality'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot decision boundary (using two features for visualization)
feature_1 = 'alcohol'
feature_2 = 'volatile acidity'

x_min, x_max = data[feature_1].min() - 1, data[feature_1].max() + 1
y_min, y_max = data[feature_2].min() - 1, data[feature_2].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Create a mesh of points to predict
mesh_points = np.zeros((xx.ravel().shape[0], X.shape[1]))
mesh_points[:, data.columns.get_loc(feature_1)] = xx.ravel()
mesh_points[:, data.columns.get_loc(feature_2)] = yy.ravel()

# Make predictions using the trained model
Z = model.predict(scaler.transform(mesh_points))
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')
plt.scatter(data[feature_1], data[feature_2], c=y, edgecolor='k', marker='o', label='Data')
plt.xlabel('Alcohol')
plt.ylabel('Volatile Acidity')
plt.title('Logistic Regression Classification with Wine Quality Dataset')
plt.colorbar(label='Quality (0: Bad, 1: Good)')
plt.show()
