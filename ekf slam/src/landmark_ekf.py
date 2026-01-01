import numpy as np

def update_landmarks_lidar(mu, Sigma, lidar_measurements, R):
    """
    EKF landmark update using LiDAR measurements.
    lidar_measurements: (M,3) absolute positions from LiDAR
    """
    M = lidar_measurements.shape[0]
    for i in range(M):
        lm_idx = 6 + 3*i
        z = lidar_measurements[i]

        # Predicted landmark
        pred = mu[lm_idx:lm_idx+3]

        # Innovation
        y = z - pred

        # Measurement Jacobian
        H = np.zeros((3, mu.shape[0]))
        H[:, lm_idx:lm_idx+3] = np.eye(3)

        # Innovation covariance
        S = H @ Sigma @ H.T + R

        # Kalman gain
        K = Sigma @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        mu += K @ y
        Sigma = (np.eye(len(mu)) - K @ H) @ Sigma

    return mu, Sigma
