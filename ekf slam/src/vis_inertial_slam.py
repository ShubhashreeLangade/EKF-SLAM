import numpy as np
from imu_ekf import predict_state
from landmark_ekf import update_landmarks_lidar

class VisualInertialSLAM2D:
    def __init__(self, num_landmarks, Q, R, dt):
        self.M = num_landmarks
        self.dt = dt
        self.state_dim = 3 + 2*self.M

        self.mu = np.zeros(self.state_dim)
        self.Sigma = np.eye(self.state_dim) * 0.01

        self.Q = Q
        self.R = R
        self.history = []

    def step(self, v, w, lidar):
        self.mu, self.Sigma = predict_state(
            self.mu, self.Sigma, v, w, self.Q, self.dt
        )

        self.mu, self.Sigma = update_landmarks_lidar(
            self.mu, self.Sigma, lidar, self.R
        )

        self.history.append(self.mu.copy())

    def trajectory(self):
        return np.array(self.history)[:, :2]

    def landmarks(self):
        return np.array([
            self.mu[3+2*i:3+2*i+2]
            for i in range(self.M)
        ])
