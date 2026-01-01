import numpy as np

def predict_state(mu, Sigma, linear_vel, angular_vel, Q, dt):
    """
    EKF prediction step using IMU data.
    mu: current state [x,y,z,roll,pitch,yaw,...landmarks]
    Sigma: covariance matrix
    linear_vel: linear velocity (3,)
    angular_vel: angular velocity (3,)
    Q: process noise (6x6)
    dt: timestep
    """
    # Predict robot position and orientation
    mu[0:3] += linear_vel * dt
    mu[3:6] += angular_vel * dt

    # Add process noise to robot part of covariance
    Sigma[0:6, 0:6] += Q
    return mu, Sigma
