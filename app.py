# app.py

import gradio as gr
import numpy as np
import tensorflow as tf

# Map dropdown labels to actual filenames
MODEL_PATHS = {
    "Simple RNN": "model_SimpleRNN (1).keras",
    "GRU": "model_GRU (1).keras",
    "Best Model (H5)": "best_model (1).h5"
}

# Load models once to avoid reload delays
LOADED_MODELS = {name: tf.keras.models.load_model(path) for name, path in MODEL_PATHS.items()}

# Define prediction function
def predict_rnn(input_text, model_choice):
    try:
        model = LOADED_MODELS[model_choice]
        arr = np.array([float(x) for x in input_text.split(",")]).reshape(1, -1, 1)
        pred = model.predict(arr)
        return str(pred)
    except Exception as e:
        return f"Error: {e}"

# Gradio interface
iface = gr.Interface(
    fn=predict_rnn,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter values like 0.1, 0.2, 0.3"),
        gr.Dropdown(choices=list(MODEL_PATHS.keys()), label="Select Model", value="Simple RNN")
    ],
    outputs="text",
    title="Live RNN Model Ensemble Tester",
    description="Input a comma-separated sequence of floats and select a model to test predictions."
)

iface.launch()
