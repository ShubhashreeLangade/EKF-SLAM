# motion_model.py
def motion_model(mu, linear_vel, angular_vel, dt):
    # Predict next state
    mu[:3] += linear_vel * dt
    mu[3:] += angular_vel * dt
    return mu
