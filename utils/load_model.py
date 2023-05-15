import os
import joblib

def load_model():
    # Load the model
    model_name = "penguins_pipe.pkl"
    absolute_path = os.path.dirname(__file__)
    relative_path = f"{model_name}"
    full_path = os.path.join(absolute_path, relative_path)
    model = joblib.load(full_path)
    return model