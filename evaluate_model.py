import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report

# Checking if the command-line argument is provided
if len(sys.argv) < 2:
    print("Usage: python evaluate_model.py <path_to_dataset_folder>")
    sys.exit(1)

# Define the path to the dataset folder from the command-line argument
dataset_folder = sys.argv[1]

# Check if the dataset folder exists
if not os.path.exists(dataset_folder):
    print(f"The provided dataset folder '{dataset_folder}' does not exist.")
    sys.exit(1)

# Load the trained model
model = keras.models.load_model('model_nitex_task.h5')

# Load and preprocess the data
data = np.load(os.path.join(dataset_folder, 'data.npy'))
labels = np.load(os.path.join(dataset_folder, 'labels.npy'))

# Evaluate the model on the dataset
test_loss, test_accuracy = model.evaluate(data, labels)

# Generate a classification report
predictions = model.predict(data)
predicted_labels = np.argmax(predictions, axis=1)
class_names = ['T-shirt', 'Trousers', 'Pullovers', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
report = classification_report(np.argmax(labels, axis=1), predicted_labels, target_names=class_names)

# Output file path
output_file = os.path.join(dataset_folder, 'output.txt')

# Write model architecture summary and evaluation metrics to the output file
with open(output_file, 'w') as file:
    file.write("Model's Architecture Summary:\n")
    model.summary(print_fn=lambda x: file.write(x + '\n'))
    file.write("\nEvaluation Metrics:\n")
    file.write(f"Test Loss: {test_loss}\n")
    file.write(f"Test Accuracy: {test_accuracy}\n")
    file.write("\nClassification Report:\n")
    file.write(report)

print(f"Model evaluation completed. Results saved in '{output_file}'.")
