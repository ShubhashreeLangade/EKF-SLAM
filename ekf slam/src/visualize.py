import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory_with_imu(mu_history, linear_vel, dataset_number):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(mu_history[:,0], mu_history[:,1], mu_history[:,2], 'r-', label='Trajectory')
    N = len(linear_vel)
    for t in range(0, N, 2):
        ax.quiver(mu_history[t,0], mu_history[t,1], mu_history[t,2],
                  linear_vel[t,0], linear_vel[t,1], linear_vel[t,2],
                  color='b', length=0.5, normalize=True)
    ax.scatter(mu_history[0,0], mu_history[0,1], mu_history[0,2], c='g', s=50, label='Start')
    ax.scatter(mu_history[-1,0], mu_history[-1,1], mu_history[-1,2], c='k', s=50, label='End')
    ax.set_title(f'EKF-SLAM: Trajectory with IMU - Dataset {dataset_number:02d}')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()

def plot_trajectory(mu_history, dataset_number):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(mu_history[:,0], mu_history[:,1], mu_history[:,2], 'b-', label='Trajectory')
    ax.scatter(mu_history[0,0], mu_history[0,1], mu_history[0,2], c='g', s=50, label='Start')
    ax.scatter(mu_history[-1,0], mu_history[-1,1], mu_history[-1,2], c='r', s=50, label='End')
    ax.set_title(f'EKF-SLAM: Trajectory - Dataset {dataset_number:02d}')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()

def plot_slam_vs_landmarks(mu_history, original_landmarks, slam_landmarks, dataset_number):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(mu_history[:,0], mu_history[:,1], mu_history[:,2], 'r-', label='Trajectory')
    ax.scatter(original_landmarks[:,0], original_landmarks[:,1], original_landmarks[:,2],
               c='g', marker='o', s=50, label='Original Landmarks')
    ax.scatter(slam_landmarks[:,0], slam_landmarks[:,1], slam_landmarks[:,2],
               c='r', marker='x', s=50, label='SLAM Landmarks')
    ax.set_title(f'EKF-SLAM: SLAM vs Landmarks - Dataset {dataset_number:02d}')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
