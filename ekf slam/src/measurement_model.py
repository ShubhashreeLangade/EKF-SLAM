# measurement_model.py
def measurement_model(mu, landmarks):
    # Predict measurements to landmarks (range, bearing)
    return landmarks - mu[:3]
