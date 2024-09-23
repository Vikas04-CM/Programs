import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load the Wine Quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Features and target variable
X = data.drop('quality', axis=1)  # Features
y = (data['quality'] > 5).astype(int)  # Target variable (1 for good quality, 0 for bad quality)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and fit the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate accuracy and ROC-AUC
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f'Accuracy: {accuracy:.2f}')
print(f'ROC AUC: {roc_auc:.2f}')

# Print confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Bad Quality', 'Good Quality']))

# Plotting (Optional)
# For visualization, we can plot two features, e.g., alcohol and acidity
plt.scatter(data['alcohol'], data['volatile acidity'], c=y, cmap='coolwarm', edgecolor='k', s=20)
plt.xlabel('Alcohol Content')
plt.ylabel('Volatile Acidity')
plt.title('Wine Quality Data (Alcohol vs. Volatile Acidity)')
plt.colorbar(label='Quality (Good or Bad)')
plt.show()
