import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data (convert pixel values from 0-255 to 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Split the training set into a smaller training set and a validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Build the classification model
model = Sequential([
    Flatten(input_shape=(28, 28)),          # Flatten the 28x28 images into 1D vectors
    Dense(128, activation='relu'),          # Hidden layer with 128 units and ReLU activation
    Dropout(0.2),                           # Dropout layer to prevent overfitting
    Dense(10, activation='softmax')         # Output layer with 10 units for 10 classes (0-9 digits)
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# TensorBoard setup
log_dir = os.path.join("C:/path/to/log_directory", "mnist_logs")  # Replace with your desired log path
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
model.fit(x_train, y_train, 
          epochs=10, 
          validation_data=(x_val, y_val), 
          callbacks=[tensorboard_callback])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# Predictions for test data
y_pred = model.predict(x_test)
y_pred_classes = tf.argmax(y_pred, axis=1).numpy()  # Get predicted class labels

# Count correct and incorrect classifications
correct = (y_pred_classes == y_test).sum()
incorrect = (y_pred_classes != y_test).sum()

# Pie chart data
labels = 'Correct', 'Incorrect'
sizes = [correct, incorrect]
colors = ['#4CAF50', '#F44336']  # Green for correct, red for incorrect
explode = (0.1, 0)  # Explode the 'Correct' slice

# Plot the pie chart
plt.figure(figsize=(7, 7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
plt.title('Classification Results')
plt.show()

# Save the trained model
model.save('mnist_classification_model.h5')
