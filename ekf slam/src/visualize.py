import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# 1️⃣ 3D Trajectory with IMU velocity vectors
# -------------------------------------------------
def plot_trajectory_with_imu(mu_history, linear_vel, dataset_number):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(
        mu_history[:,0],
        mu_history[:,1],
        mu_history[:,2],
        'r-', label='Trajectory'
    )

    for t in range(0, len(mu_history), 3):
        ax.quiver(
            mu_history[t,0], mu_history[t,1], mu_history[t,2],
            linear_vel[t,0], linear_vel[t,1], linear_vel[t,2],
            color='b', length=0.5, normalize=True
        )

    ax.scatter(mu_history[0,0], mu_history[0,1], mu_history[0,2],
               c='g', s=60, label='Start')
    ax.scatter(mu_history[-1,0], mu_history[-1,1], mu_history[-1,2],
               c='k', s=60, label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Trajectory with IMU (Dataset {dataset_number:02d})')
    ax.legend()


# -------------------------------------------------
# 2️⃣ 3D Trajectory only
# -------------------------------------------------
def plot_trajectory(mu_history, dataset_number):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(
        mu_history[:,0],
        mu_history[:,1],
        mu_history[:,2],
        'b-', label='Trajectory'
    )

    ax.scatter(mu_history[0,0], mu_history[0,1], mu_history[0,2],
               c='g', s=60, label='Start')
    ax.scatter(mu_history[-1,0], mu_history[-1,1], mu_history[-1,2],
               c='r', s=60, label='End')

    ax.set_title(f'Trajectory Only (Dataset {dataset_number:02d})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()


# -------------------------------------------------
# 3️⃣ SLAM vs Original Landmarks (3D)
# -------------------------------------------------
def plot_slam_vs_landmarks(mu_history, original_landmarks, slam_landmarks, dataset_number):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(
        mu_history[:,0],
        mu_history[:,1],
        mu_history[:,2],
        'r-', label='SLAM Trajectory'
    )

    ax.scatter(
        original_landmarks[:,0],
        original_landmarks[:,1],
        original_landmarks[:,2],
        c='g', s=40, label='Original Landmarks'
    )

    ax.scatter(
        slam_landmarks[:,0],
        slam_landmarks[:,1],
        slam_landmarks[:,2],
        c='r', marker='x', s=60, label='SLAM Landmarks'
    )

    ax.set_title(f'SLAM vs Landmarks (Dataset {dataset_number:02d})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()


# -------------------------------------------------
# 4️⃣ 2D EKF-SLAM Estimated Trajectory (LIKE YOUR IMAGE)
# -------------------------------------------------
def plot_ekf_slam_2d(mu_history, slam_landmarks, dataset_number):
    traj = mu_history[:, :2]

    plt.figure(figsize=(8,6))
    plt.plot(traj[:,0], traj[:,1], 'r-', linewidth=2, label='EKF Trajectory')

    plt.scatter(
        slam_landmarks[:,0],
        slam_landmarks[:,1],
        c='g', s=10, label='Landmarks'
    )

    plt.scatter(traj[0,0], traj[0,1],
                c='b', s=80, marker='s', label='Start')
    plt.scatter(traj[-1,0], traj[-1,1],
                c='orange', s=80, marker='o', label='End')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Estimated Trajectory with EKF Landmark Mapping')
    plt.legend()
    plt.grid(True)
