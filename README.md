CIFAR-10 Image Classification with Custom Neural Network
This project uses TensorFlow and Keras to build a neural network model for classifying images from the CIFAR-10 dataset. It explores various neural network architectures and optimization techniques to achieve the best possible accuracy. Additionally, the code evaluates the performance of different configurations and visualizes the results through plots and confusion matrices.

Table of Contents
Installation Requirements
Dataset
Model Architecture
Training and Evaluation
Results
Confusion Matrix
Comparing Loss Functions
MNIST Recommendations
Installation Requirements
Before running the code, you need to install the required libraries. You can do so by running the following command:

bash
Copy
pip install tensorflow numpy matplotlib seaborn scikit-learn
Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck). These images are split into 50,000 training images and 10,000 test images.

Data Preprocessing:
The pixel values of the images are normalized to the range [0, 1].
The training data is split, with 10% reserved for validation.
Class labels are converted to one-hot encoded vectors for training the neural network.
Model Architecture
The neural network model is built using Keras and TensorFlow. The architecture is flexible and can be customized by modifying the number of layers, units per layer, activation functions, and kernel initializers.

Model Parameters:
Input Layer: Flattened 32x32x3 image.
Hidden Layers: Can have customizable number of layers and units. Common activation functions include 'ReLU', 'Sigmoid'.
Output Layer: Softmax activation with 10 units (one for each class).
Customizable Hyperparameters:
Number of layers
Number of units in each hidden layer
Activation function
Optimizer (SGD, RMSprop, Adam, Nadam)
Learning rate
Weight initializer
Training and Evaluation
The training process uses various configurations of the neural network, including different optimizers and hyperparameters. After training, the model is evaluated on the test set, and the test accuracy is reported.

The train_model() function compiles the model with the specified optimizer and loss function, and trains it on the training data for a set number of epochs. Validation data is used to monitor performance during training.

Evaluation Results:
The model's performance is evaluated on test data after each configuration is trained. The best-performing model is selected based on the test accuracy.

python
Copy
# Example output:
Config: {'layers': 4, 'units': 128, 'activation_fn': 'relu', 'opt_choice': 'adam', 'learning_rate': 0.002, 'init_method': 'glorot_normal'} 
-> Test Accuracy: 0.8412
Results
For each configuration of the model, the test accuracy is printed. The best model is identified, and its performance is reported.

python
Copy
# Best Model Test Accuracy:
Best Model Test Accuracy: 0.8500
Confusion Matrix
The confusion matrix is generated for the best-performing model to provide insight into the classification errors. This is visualized as a heatmap.

python
Copy
# Code for plotting confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', xticklabels=categories, yticklabels=categories)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
Comparing Loss Functions
The code compares the performance of the best model using two different loss functions: Cross-Entropy Loss and Mean Squared Error (MSE). The accuracy is reported for both loss functions to evaluate their effect on performance.

python
Copy
# Cross Entropy vs Mean Squared Error Loss
Cross Entropy Accuracy: 0.8500, Mean Squared Error Accuracy: 0.8650
MNIST Recommendations
In addition to CIFAR-10, the code also provides recommended configurations for training on the MNIST dataset, a well-known dataset consisting of handwritten digits.

Example configurations are suggested, such as:

5 layers, 256 units, ReLU activation, Adam optimizer, learning rate: 1.5e-3.
4 layers, 128 units, ReLU activation, SGD optimizer, learning rate: 2e-3.
These configurations are tailored for MNIST, providing an optimal starting point for similar image classification tasks.

How to Use
Load the Dataset: The CIFAR-10 dataset is automatically loaded and split into training, validation, and test sets.
Visualize Samples: The visualize_samples() function displays a grid of 10 random images with their corresponding labels.
Train and Evaluate: Train the model using different configurations. The model will be evaluated based on test accuracy.
Confusion Matrix: Once training is complete, the confusion matrix for the best model is displayed.
Compare Loss Functions: The performance with cross-entropy and MSE loss functions is compared.
MNIST Recommendations: You can use the provided configurations as a starting point for training on the MNIST dataset.
Conclusion
This script provides a flexible framework for training neural networks on CIFAR-10 with different architectures and hyperparameter settings. It also offers tools for visualizing model performance and comparing various loss functions and optimizers. The recommendations for MNIST are intended to give a strong baseline for similar classification tasks.

Note: Adjust the configurations and hyperparameters for further experimentation to achieve better results or adapt it for other image classification tasks.
