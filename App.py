import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("Handwritten Digit Recognition (0-9)")

st.write("Draw a digit (0-9) below and click 'Predict Digit' to see the model's prediction.")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict Digit"):
    if canvas_result.image_data is not None:
        # Preprocess the image
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'), mode='L')
        img = img.resize((28, 28))
        img_arr = np.array(img)
        img_arr = img_arr / 255.0  # Normalize
        img_arr = img_arr.reshape(1, 28, 28, 1)

        # Load the pretrained model
        model = tf.keras.models.load_model("mnist_cnn.h5")
        prediction = model.predict(img_arr)
        pred_digit = np.argmax(prediction)

        st.subheader(f"Predicted Digit: {pred_digit}")
        st.bar_chart(prediction[0])
    else:
        st.warning("Please draw a digit first.")

st.markdown("---")
st.write("Made by Shivam using Streamlit and TensorFlow")