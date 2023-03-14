import numpy as np
from pr3_utils import *
from robot import robot
from tqdm import tqdm
from transforms3d.euler import euler2mat
import sys
from filter_measurements import *

def visualize_trajectory_landmark_2d(pose, landmark, path_name="Unknown",show_ori=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
		landmark: 3*N matrix
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
    ax.scatter(landmark[0,:],landmark[1,:],s=1, label="landmarks")
  
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.show(block=True)

    return fig, ax

if __name__ == '__main__':

    # Load the measurements
    filename = "../data/10.npz"
    t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

    dt = t[0,1:] - t[0,:-1]
    my_robot = robot(K, b, imu_T_cam)

    wTi = np.zeros((4,4,dt.shape[0]+1))
    wTi = np.zeros((4,4,dt.shape[0]+1))
    wTi[:,:,0] = np.eye(4)

    n_landmarks = 1000
    sub_features = disparity_filter_measurements(features, n=5000)

    # sub_features = random_subsample_measurements(features, n=5000)
    landmarks = None

    # (a) IMU Localization via EKF Prediction
    for i in tqdm(range(dt.shape[0])):
        wTi[:,:,i+1] = my_robot.predict_pose(dt[i], wTi[:,:,i], linear_velocity[:,i], angular_velocity[:,i])

        # (b) predict feature location using dead reckoning
        imu_mloc = my_robot.get_pos_from_stereo(sub_features[:,:,i+1])

        w_T_mloc = (wTi[:,:,i+1] @ imu_mloc)
        w_T_mloc /= w_T_mloc[3,:]        

        if landmarks is None:
            landmarks = w_T_mloc[:3,:]
        else:
            landmarks = np.hstack((landmarks,w_T_mloc[:3,:]))

    landmarks = np.array(landmarks)
    # print(landmarks.shape)

    # (b) Landmark Mapping via EKF Update

    # (c) Visual-Inertial SLAM

    # You can use the function below to visualize the robot pose over time
    # visualize_trajectory_2d(wTi, show_ori = True)
    visualize_trajectory_landmark_2d(wTi, landmarks, path_name="dead_reckoning",show_ori=True)
