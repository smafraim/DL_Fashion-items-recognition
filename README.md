# Fashion Items Recognition with Deep Learning

----
## Overview

This project focuses on building a deep learning model to recognize various fashion items. I've used the Fashion MNIST dataset from keras for training and evaluation. The goal is to develop a model that can accurately classify clothing items like T-shirts, trousers, dresses, etc.

## Approach

### 1. Data Preparation
- Loaded and preprocessed the Fashion MNIST dataset.
- The dataset includes grayscale images of fashion items, each labeled with a corresponding class from a total of 10 classes.

### 2. Model Architecture
- Designed a Convolutional Neural Network (CNN) for image classification.
- The CNN consists of 2 convolutional layers, 1 max-pooling layers, and several fully connected layers.
- Utilized the ReLU activation function and softmax activation for the output layer.

### 3. Training
- Compiled the model with an appropriate optimizer named `adam optimizer`, loss function named `sparse_categorical_crossentropy`, and `accuracy` from metrics.
- The model is trained on the training data with 7 epochs.

### 4. Evaluation & Analysis
- Evaluated the trained model on the test data and calculated classification accuracy that consisted of various matrices.
- Generated a confusion matrix, a heatmap and a classification report for a more detailed analysis.

### 5. Saving and Deployment
- The trained model is saved to a file  `model_nitex_task.h5` for future use especially for the evaluation file named `evaluation_model.py`
- The instructions for deploying the model and making predictions are given below:

## Instructions

### Prerequisites

- Python 3.x
- TensorFlow (install with `pip install tensorflow`)
- NumPy (install with `pip install numpy`)
- Pandas (install with `pip install pandas`)
- Matplotlib (install with `pip install matplotlib`)
- Seaborn (install with `pip install seaborn`)

### Running the Code

1. Clone or download this repository to your local machine.

2. Open a command prompt or terminal.

3. Navigate to the project directory where the code is located.

4. Run the evaluation script with the path to the data folder as an argument:

   ```
   python evaluate_model.py data
   ```

   Replace `data` with the actual path to the folder of yours if you're executing that contains the dataset for evaluation as in my case the directory is only `data`

5. The script will load the trained model, evaluate it on the provided dataset, and generate an `output.txt` file with the model's architecture summary, evaluation metrics and other insights such as the classification report as well.

6. The results can be analyzed in the `output.txt` file.

### Additional Notes

- The `Fashion MNIST - NITEX.ipynb` is a jupyter notebook file that contains a detailed description of whatever I implemented based on the `Fashion Items Recognition with Deep Learning`

- The project uses the Fashion MNIST dataset, but you can adapt it for other image classification tasks like taking anything inside NITEX for maintaining the `Human-in-the-loop` strategy.

- Feel free to explore and experiment with the model architecture and hyperparameters to improve performance.

- The saved model can be deployed for making predictions in a variety of applications.

## Contributors

- Syed Md. Afraim
-     syedmohammadafraim@gmail.com

Feel free to reach out with any questions or feedback!

---
