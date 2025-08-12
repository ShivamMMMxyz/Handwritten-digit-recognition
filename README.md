MNIST Strong CNN Classifier with Streamlit UI
This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model is trained using TensorFlow and Keras and achieves high accuracy after 30 epochs. Additionally, a Streamlit-based user interface is included to allow easy interaction and digit recognition via a web app.

Features
Preprocesses and normalizes the MNIST dataset for CNN input.

Builds a deep CNN architecture with Conv2D, MaxPooling, Dropout, and Dense layers.

Trains the model for 30 epochs with validation on test data.

Saves the trained model (mnist_strong_cnn.h5).

Streamlit UI for drawing digits and getting real-time predictions.

Requirements
Python 3.12

TensorFlow

Streamlit

numpy

matplotlib (optional, for visualization)

How to Run
Train the model (optional)
Run the script to train the CNN model on MNIST and save the model file:

bash
Copy
Edit
python train_mnist_cnn.py
Run Streamlit UI
Launch the Streamlit app to interact with the trained model:

bash
Copy
Edit
streamlit run app.py
Using the App

Draw a digit in the canvas.

Click the "Predict" button to see the model's classification result.

Files
train_mnist_cnn.py — Script containing model training and saving code.

mnist_strong_cnn.h5 — Saved Keras model file (generated after training).

app.py — Streamlit UI code for digit input and prediction.

Model Architecture Summary
Conv2D (32 filters, 3x3) + ReLU

Conv2D (64 filters, 3x3) + ReLU

MaxPooling (2x2)

Dropout (0.25)

Flatten

Dense (128 units) + ReLU

Dropout (0.5)

Dense (10 units) + Softmax

Notes
Model accuracy improves significantly after 30 epochs.

Dropout layers help reduce overfitting.

The Streamlit UI provides an easy way to test the model without coding.
