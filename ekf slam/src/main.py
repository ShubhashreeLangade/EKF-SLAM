import os
import numpy as np
from imu_ekf import predict_state
from landmark_ekf import update_landmarks_lidar
from visualize import plot_trajectory_with_imu, plot_trajectory, plot_slam_vs_landmarks
import matplotlib.pyplot as plt

# ---------------- Dataset Setup ----------------
dataset_number = 11
data_dir = '../data/'
data_file = os.path.join(data_dir, f'{dataset_number:02d}.npz')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Generate fake dataset if missing
if not os.path.exists(data_file):
    N = 50
    M = 5
    linear_velocity = np.random.rand(N,3)*0.1
    angular_velocity = np.random.rand(N,3)*0.01
    features = np.random.rand(N,M,4)*10  # x,y,z,intensity
    np.savez(data_file, linear_velocity=linear_velocity,
             angular_velocity=angular_velocity,
             features=features)
    print(f"Fake dataset {data_file} created!")

# ---------------- Load Dataset ----------------
data = np.load(data_file)
linear_vel = data['linear_velocity']
angular_vel = data['angular_velocity']
features = data['features']
N, M, _ = features.shape

# ---------------- EKF-SLAM Initialization ----------------
state_dim = 6 + 3*M
mu = np.zeros(state_dim)
Sigma = np.eye(state_dim) * 0.01
Q = np.eye(6)*0.0001
R = np.eye(3)*0.05
dt = 0.01
mu_history = []

# ---------------- EKF-SLAM Loop ----------------
for t in range(N):
    # 1️⃣ Prediction using IMU
    mu, Sigma = predict_state(mu, Sigma, linear_vel[t], angular_vel[t], Q, dt)
    
    # 2️⃣ Simulated LiDAR measurement (with noise)
    lidar_measurements = features[t,:,:3] + np.random.randn(M,3)*0.05
    
    # 3️⃣ Update landmarks using LiDAR
    mu, Sigma = update_landmarks_lidar(mu, Sigma, lidar_measurements, R)
    
    mu_history.append(mu.copy())

mu_history = np.array(mu_history)
slam_landmarks = np.array([mu[6+3*i:6+3*i+3] for i in range(M)])

# ---------------- Visualization ----------------
# Create all figures first, then show them together
plot_trajectory_with_imu(mu_history, linear_vel, dataset_number)
plot_trajectory(mu_history, dataset_number)
plot_slam_vs_landmarks(mu_history, features[0,:,:3], slam_landmarks, dataset_number)

# Show all figures at once
plt.show(block=True)   # block=True ensures all figures stay open
