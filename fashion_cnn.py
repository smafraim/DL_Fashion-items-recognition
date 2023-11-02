import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load the Fashion MNIST dataset
f_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = f_mnist.load_data()

class_names = ['T-shirt', 'Trousers', 'Pullovers', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# Data Exploration
print("Train Images Shape:", train_images.shape)
print("Test Images Shape:", test_images.shape)
print("Train Labels Shape:", train_labels.shape)

# Data Preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build a CNN Model
model = Sequential([
    Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(),
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

# Compile the Model
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=adam_optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fitting the Model
model.fit(train_images, train_labels, epochs=5)

# Accuracy Evaluation
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy of the model is {test_accuracy} or {round(test_accuracy*100, 2)}%")

# Make Predictions on Test Images
preds = model.predict(test_images)

# Visualization
plt.figure(figsize=(12, 12))
for val in range(36):
    plt.subplot(6, 6, val + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[val], cmap=plt.cm.binary)
    pred_label = np.argmax(preds[val])
    true_label = test_labels[val]

    if pred_label == true_label:
        text_color = 'green'
    else:
        text_color = 'red'

    plt.xlabel(f"{class_names[pred_label]} ({class_names[true_label]})", color=text_color)

plt.show()

# Confusion Matrix
matrix_pred = [np.argmax(label) for label in preds]
conf_matrix = confusion_matrix(test_labels, matrix_pred)

# Visualization
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, cmap='inferno', annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)

# Classification Report
report = classification_report(test_labels, matrix_pred, target_names=class_names)
print(report)

# Saving the Model
model.save('model_nitex_task.h5')

# Load the Deployed Model
deploy = keras.models.load_model('model_nitex_task.h5')

# Make Predictions with Deployed Model
pred_after_save = deploy.predict(test_images).round(2)
np.argmax(pred_after_save)
