# app.py

import gradio as gr
import numpy as np
import tensorflow as tf

# Load the model (choose one of the models you want to test)
model = tf.keras.models.load_model("best_model (2).h5")  # Example: switch between .keras or .h5

# Define prediction function
def predict_rnn(input_text):
    try:
        # Convert comma-separated string to array
        arr = np.array([float(x) for x in input_text.split(",")]).reshape(1, -1, 1)
        pred = model.predict(arr)
        return str(pred)
    except Exception as e:
        return f"Error: {e}"

# Define Gradio interface
iface = gr.Interface(
    fn=predict_rnn,
    inputs=gr.Textbox(lines=2, placeholder="Enter values like 0.1, 0.2, 0.3"),
    outputs="text",
    title="Live RNN Model Predictor",
    description="Input a comma-separated sequence of floats."
)

# Launch interface
iface.launch()
