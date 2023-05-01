import numpy as np
from pr3_utils import *
from robot import robot
from tqdm import tqdm
from transforms3d.euler import euler2mat
import sys
from filter_measurements import *
from kalman_filter import *
import cv2
import os

if __name__ == '__main__':
    # Load the measurements
    filename = "../data/10.npz"
    t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

    imu_T_cam[:3,:3] = euler2mat(np.pi,0,0, 'sxyz') @ imu_T_cam[:3,:3] 

    dt = t[0,1:] - t[0,:-1]
    my_robot = robot(K, b, imu_T_cam)

    wTi = np.zeros((4,4,dt.shape[0]+1))
    wTi[:,:,0] = np.eye(4)

    sub_features = disparity_filter_measurements(features, n=2000)
    # sub_features = random_subsample_measurements(features, n=600)
    n_landmarks = sub_features.shape[1]
    kf = kalman_filter()

    m_state = np.ones((3*n_landmarks,1)) * 0
    new_idx = sub_features[0,:,0] > 0
    w_T_mloc = wTi[:, :, 0] @ my_robot.get_pos_from_stereo(sub_features[:,:,0])
    w_T_mloc /= w_T_mloc[3,:]
    w_T_mloc = w_T_mloc[:3,:]

    init_list_idx = np.arange(n_landmarks)[sub_features[0, :, 0] > 0]
    sigma = np.eye(3*n_landmarks+6) * 1
    m_state, sigma = kf.init_landmarks(m_state, sigma, w_T_mloc, sub_features[:,:,0], init_list_idx)
    visited_landmarks = set(np.arange(n_landmarks)[sub_features[0, :, 0] > 0])

    np.set_printoptions(2)

    # Camera img visualization
    image_dir = "../data/images_" + filename.split('/')[-1].split('.')[0]
    images = sorted(os.listdir(image_dir))

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1408, 376)

    landmarks = None
    for i in tqdm(range(dt.shape[0])):
        # (a) IMU Localization using dead reckoning
        wTi[:,:,i+1], sigma = kf.motion_prediction( my_robot,
                                                    dt[i], 
                                                    wTi[:,:,i], 
                                                    linear_velocity[:,i], 
                                                    angular_velocity[:,i], sigma)

        if i%10 == 0:
            img = cv2.imread(os.path.join(image_dir, images[i//10]))
            for j in range(sub_features.shape[1]):
                if sub_features[0,j,i+1] > 0:
                    cv2.circle(img, tuple(sub_features[:2,j,i+1].astype(int)), 1, (0,200,255), 8)
            cv2.imshow('img', img)
            cv2.waitKey(100)

        # (b) predict feature location using dead reckoning
        imu_mloc = my_robot.get_pos_from_stereo(sub_features[:,:,i+1])
        w_T_mloc = (wTi[:,:,i+1] @ imu_mloc)
        w_T_mloc /= w_T_mloc[3,:]

        obs_landmarks = set(np.arange(n_landmarks)[sub_features[0, :, i+1] > 0])
        update_idx_set = visited_landmarks & obs_landmarks
        init_list_idx = list(obs_landmarks - update_idx_set)
        update_list_idx = list(update_idx_set)

        measurements = sub_features[:, update_list_idx, i+1]

        if len(init_list_idx) > 0:
            kf.init_landmarks(m_state, sigma, w_T_mloc[:3,:], sub_features[:,:, i+1], init_list_idx)
            visited_landmarks = visited_landmarks.union(set(init_list_idx))

        # m_state, sigma = kf.map_only_update(my_robot, 
        #                                     sigma, 
        #                                     imu_T_cam, 
        #                                     wTi[:,:, i+1], 
        #                                     m_state, 
        #                                     measurements, 
        #                                     n_landmarks, 
        #                                     update_list_idx)

        wTi[:,:, i+1], m_state, sigma = kf.update_full_state(my_robot,
                                                        wTi[:,:,i+1],
                                                        sigma,
                                                        m_state,
                                                        measurements,
                                                        n_landmarks,
                                                        update_list_idx)

        m = m_state.T.reshape(-1,3).T

        if i % 1 == 0:
            visualize_trajectory_landmark_2d(wTi, m, path_name="visual SLAM",show_ori=True, save_path="results_across_time/plots/{}.png".format(str(i).zfill(6)), show=False, max_num=i)
        cv2.imwrite("results_across_time/images/{}.png".format(str(i).zfill(6)), img)
    visualize_trajectory_landmark_2d(wTi, m, path_name="Visual mapping",show_ori=True)

