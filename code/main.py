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

    dt = t[0,1:] - t[0,:-1]
    my_robot = robot(K, b, imu_T_cam)

    wTi = np.zeros((4,4,dt.shape[0]+1))
    wTi = np.zeros((4,4,dt.shape[0]+1))
    wTi[:,:,0] = np.eye(4)

    sub_features = disparity_filter_measurements(features, n=1000)
    n_landmarks = sub_features.shape[1]
    kf = kalman_filter()

    m_state = np.ones((3*n_landmarks,1)) * -5000
    new_idx = sub_features[0,:,0] > 0
    w_T_mloc = wTi[:, :, 0] @ my_robot.get_pos_from_stereo(sub_features[:,:,0])
    w_T_mloc /= w_T_mloc[3,:]
    w_T_mloc = w_T_mloc[:3,:]

    sigma = np.zeros((3*n_landmarks, 3*n_landmarks))
    m_state, sigma = init_landmarks(m_state, sigma, w_T_mloc, sub_features[:,:,0])

    # (a) IMU Localization using dead reckoning
    landmarks = None
    new_update_idx = None
    old_update_idx = sub_features[0,:,0] > 0
    for i in tqdm(range(10)):#dt.shape[0])):
        wTi[:,:,i+1] = my_robot.predict_pose(dt[i], wTi[:,:,i], linear_velocity[:,i], 
                                             angular_velocity[:,i])

        # (b) predict feature location using dead reckoning
        imu_mloc = my_robot.get_pos_from_stereo(sub_features[:,:,i+1])
        w_T_mloc = (wTi[:,:,i+1] @ imu_mloc)
        w_T_mloc /= w_T_mloc[3,:]

        new_update_idx = sub_features[0,:,i+1] > 0
        if not np.array_equal(new_update_idx, old_update_idx):
            print(np.arange(new_update_idx.shape[0])[new_update_idx])
            print(np.arange(old_update_idx.shape[0])[old_update_idx])
            print("detected new landmarks")
            init_landmarks(m_state, sigma, w_T_mloc[:3,:], sub_features[:,:, i+1])

        old_update_idx = np.copy(new_update_idx)
        measurements = sub_features[:, new_update_idx, i+1]

        m_state, sigma = kf.update(my_robot, sigma, imu_T_cam, wTi[:,:, i+1], 
                            m_state, measurements, n_landmarks, new_update_idx)

        # if landmarks is None:
        #     landmarks = w_T_mloc[:3,:]
        # else:
        #     landmarks = np.hstack((landmarks,w_T_mloc[:3,:]))

    # landmarks = np.array(landmarks)
    m = m_state.T.reshape(3,-1).T
    visualize_trajectory_landmark_2d(wTi, m, path_name="dead_reckoning",show_ori=True)

    # (b) Landmark Mapping via EKF Update

    # (c) Visual-Inertial SLAM
