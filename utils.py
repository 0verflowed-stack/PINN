import numpy as np

def calculate_max_relative_error(u_pred, u_exact):
    return 100 * np.linalg.norm(u_exact - u_pred, np.inf) / np.linalg.norm(u_exact, np.inf)

def save_and_load_weights(modelClass, model, layers, fileName):
    pass