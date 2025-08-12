# ==========================
# 1. Import Libraries
# ==========================
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# ==========================
# 2. Load and Normalize Data
# ==========================
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0   # Normalize to [0,1]
x_test = x_test.astype('float32') / 255.0

# Reshape for CNN input
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ==========================
# 3. Build Stronger CNN Model
# ==========================
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ==========================
# 4. Train Model (30 Epochs)
# ==========================
history = model.fit( 
    x_train, y_train,
    batch_size=128,
    epochs=30,
    validation_data=(x_test, y_test),
    verbose=2
)

# ==========================
# 5. Save Model
# ==========================
model.save('mnist_strong_cnn.h5')
print("Model saved as mnist_strong_cnn.h5")

# ==========================
# 6. Download Model from Colab
# ==========================
from google.colab import files
files.download('mnist_strong_cnn.h5')