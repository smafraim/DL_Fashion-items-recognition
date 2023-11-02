import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_fashion_mnist_test(data_folder):
    test_data = np.loadtxt(os.path.join(data_folder, "fashion-mnist_test.csv"), delimiter=',', skiprows=1)
    test_images = test_data[:, 1:] / 255.0  # Normalize pixel values
    test_images = test_images.reshape(-1, 28, 28, 1)
    test_labels = test_data[:, 0]  # The first column is the labels
    return test_images, test_labels


def load_image_data(data_folder, filename):
    file_path = os.path.join(data_folder, filename)
    with open(file_path, 'rb') as file:
        file_contents = file.read()
    image_data = np.frombuffer(file_contents, np.uint8, offset=16)
    return image_data


def evaluate_fashion_mnist_model(model, data_folder):
    try:
        test_images, test_labels = load_fashion_mnist_test(data_folder)
    except FileNotFoundError:
        print("Error: The provided data folder does not exist.")
        sys.exit(1)

    # Evaluating the model on the test data
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)

    # Generating predictions on the test data
    predictions = model.predict(test_images)

    # Calculating the classification accuracy
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(test_labels, predicted_labels)

    return test_accuracy, accuracy


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python evaluate_model.py <data_folder>")
        sys.exit(1)

    data_folder = sys.argv[1]

    if not os.path.exists(data_folder):
        print(f"Error: The data folder '{data_folder}' does not exist.")
        sys.exit(1)

    # Loading trained model
    model = keras.models.load_model('model_nitex_task.h5')

    # Loading the test images
    filename = 't10k-images-idx3-ubyte'
    image_data_test = load_image_data(data_folder, filename)

    # Evaluating the model
    test_accuracy, accuracy = evaluate_fashion_mnist_model(model, data_folder)

    # Generating the output.txt file
    with open('output.txt', 'w') as output_file:
        output_file.write("Model's Architecture Summary:\n")
        model.summary(print_fn=lambda x: output_file.write(x + '\n'))
        output_file.write(f"Test Accuracy: {test_accuracy} or {round(test_accuracy*100,2)}%\n")
        output_file.write(f"Classification Accuracy: {accuracy} or {round(accuracy*100,2)}%\n")
        output_file.write("Additional insights or observations go here.")

    print("Evaluation complete. Results saved in output.txt.")
