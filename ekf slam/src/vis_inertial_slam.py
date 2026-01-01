from imu_ekf import imu_predict
from landmark_ekf import landmark_update

def run_vi_slam(mu, Sigma, landmarks, linear_vel, angular_vel, features, Q, R):
    """
    Full Visual-Inertial SLAM loop
    """
    mu_history = []

    for t in range(len(linear_vel)):
        # IMU prediction
        mu, Sigma = imu_predict(mu, Sigma, linear_vel[t], angular_vel[t], Q)

        # Landmark update
        if t < features.shape[0]:
            observations = features[t][:, :3]  # x, y, z only
            landmarks = landmark_update(mu, Sigma, landmarks, observations, R)

        mu_history.append(mu.copy())

    return mu_history, landmarks
