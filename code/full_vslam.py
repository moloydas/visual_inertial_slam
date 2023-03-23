#full_vslam.py

import numpy as np
from pr3_utils import *
from robot import robot
from tqdm import tqdm
from transforms3d.euler import euler2mat
import sys
from filter_measurements import *
from kalman_filter import *

if __name__ == '__main__':
    # Load the measurements
    filename = "../data/10.npz"
    t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

    imu_T_cam[:3,:3] = imu_T_cam[:3,:3] @ euler2mat(180,0,0, 'sxyz')

    dt = t[0,1:] - t[0,:-1]
    my_robot = robot(K, b, imu_T_cam)

    wTi = np.zeros((4,4,dt.shape[0]+1))
    wTi[:,:,0] = np.eye(4)

    sub_features = disparity_filter_measurements(features, n=100)
    n_landmarks = sub_features.shape[1]
    kf = kalman_filter()

    state = np.zeros((3*n_landmarks,1))
    m_state = np.ones((3*n_landmarks,1)) * -5000
    new_idx = sub_features[0,:,0] > 0
    w_T_mloc = wTi[:, :, 0] @ my_robot.get_pos_from_stereo(sub_features[:,:,0])
    w_T_mloc /= w_T_mloc[3,:]
    w_T_mloc = w_T_mloc[:3,:]

    init_list_idx = np.arange(n_landmarks)[sub_features[0, :, 0] > 0]
    sigma = np.zeros((3*n_landmarks, 3*n_landmarks))
    pose_sigma = np.zeros((6,6))
    m_state, sigma = kf.init_landmarks(m_state, sigma, w_T_mloc, sub_features[:,:,0], init_list_idx)
    visited_landmarks = set(np.arange(n_landmarks)[sub_features[0, :, 0] > 0])

    # (a) IMU Localization using dead reckoning
    landmarks = None
    for i in tqdm(range(dt.shape[0])):
        wTi[:,:,i+1], pose_sigma = kf.motion_prediction(my_robot, dt[i], wTi[:,:,i], linear_velocity[:,i], 
        #                                          angular_velocity[:,i], pose_sigma)

        # (b) predict feature location using dead reckoning
        imu_mloc = my_robot.get_pos_from_stereo(sub_features[:,:,i+1])
        w_T_mloc = (wTi[:,:,i+1] @ imu_mloc)
        w_T_mloc /= w_T_mloc[3,:]

        obs_landmarks = set(np.arange(n_landmarks)[sub_features[0, :, i+1] > 0])
        update_idx_set = visited_landmarks & obs_landmarks
        init_list_idx = list(obs_landmarks - update_idx_set)
        update_list_idx = list(update_idx_set)

        measurements = sub_features[:, update_list_idx, i+1]

        m_state, sigma = kf.map_only_update(my_robot, sigma, imu_T_cam, wTi[:,:, i+1], 
                            m_state, measurements, n_landmarks, update_list_idx)

        wTi[:,:, i+1], pose_sigma = kf.update_pose(my_robot, wTi[:,:,i+1], pose_sigma, imu_T_cam, m_state, measurements, update_list_idx)

        if len(init_list_idx) > 0:
            print("detected new landmarks")
            kf.init_landmarks(m_state, sigma, w_T_mloc[:3,:], sub_features[:,:, i+1], init_list_idx)
            visited_landmarks = visited_landmarks.union(set(init_list_idx))

    m = m_state.T.reshape(-1,3).T
    visualize_trajectory_landmark_2d(wTi, m, path_name="KF map and pose",show_ori=False)

