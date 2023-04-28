#kalman_filter.py
import numpy as np
from pr3_utils import *
from robot import *
import sys 
# Kalman Filter
class kalman_filter:
    def __init__(self, vx=0.1, vz=40) -> None:
        self.vz = vz
        self.W = np.eye(6)
        self.W[:3,:3] *= 0.01
        self.W[3:,3:] *= 0.01
        self.P = np.zeros((3,4))
        self.P[:,:3] = np.eye(3)

    def motion_prediction(self, robot, dt, mu, velocity, angular_vel, sigma):
        mu_t = robot.predict_pose(dt, mu, velocity, angular_vel)
        twist = np.expand_dims(np.hstack((velocity,angular_vel)), axis=0)
        adjoint = np.squeeze(pr3_utils.pose2adpose(axangle2pose(-dt*twist)))
        sigma[:6,:6] = adjoint @ sigma[:6,:6] @ adjoint.T + self.W

        return mu_t, sigma

    def map_only_update(self, my_robot, sigma, i_T_c, w_T_i, mu_t, measurements, n_landmarks, update_idx):
        P = np.zeros((3,4))
        P[:,:3] = np.eye(3)
        nt = measurements.shape[1]
        H = np.zeros((4*nt, 3*n_landmarks))

        c_T_i = inversePose(i_T_c)
        i_T_w = inversePose(w_T_i)
        IV = np.eye(4*nt) * self.vz
        norm_mu_t = mu_t.T.reshape(-1,3).T
        norm_mu_t = np.vstack((norm_mu_t, np.ones((1, norm_mu_t.shape[1]))))

        i = 0
        for j in range(len(update_idx)):
            j_idx = update_idx[j]

            mu_j = mu_t[3*j_idx:(3*j_idx+3),0].reshape(3,1)
            m_j = np.vstack((mu_j, 1))

            q = c_T_i @ i_T_w @ m_j
            d_pi = np.squeeze(projectionJacobian(q.T))
            H[4*i:(4*i+4), 3*j_idx:(3*j_idx+3)] = my_robot.Ks @ d_pi @ c_T_i @ i_T_w @ P.T
            i += 1

        KG = sigma[6:,6:] @ H.T @ np.linalg.inv(H @ sigma[6:,6:] @ H.T + IV)
        zt = measurements.flatten(order="F")
        q = c_T_i @ i_T_w @ norm_mu_t
        pred_z = my_robot.Ks @ projection(q.T).T
        innovation = zt - pred_z[:, update_idx].flatten(order="F")
        del_mu = KG @ innovation
        mu_t1 = mu_t + del_mu.reshape(-1,1)
        sigma[6:,6:] = (np.eye(3*n_landmarks) - KG @ H) @ sigma[6:,6:]
        return mu_t1, sigma

    def update_state(self, my_robot, pose, sigma, map_pts, measurements, n_landmarks, update_idx):
        nt = measurements.shape[1]
        HR = np.zeros((4*nt, 6))
        HM = np.zeros((4*nt, 3*n_landmarks))

        c_T_i = my_robot.cTi
        i_T_w = inversePose(pose)
 
        IV = np.eye(4*nt) * self.vz
        norm_map = map_pts.T.reshape(-1,3).T
        norm_map = np.vstack((norm_map, np.ones((1, norm_map.shape[1]))))
        q = c_T_i @ i_T_w @ norm_map
        pred_z = my_robot.Ks @ projection(q.T).T
        zt = measurements.flatten(order="F")

        # create HM matrix
        i = 0
        for j in range(len(update_idx)):
            j_idx = update_idx[j]

            mu_j = map_pts[3*j_idx:(3*j_idx+3),0].reshape(3,1)
            m_j = np.vstack((mu_j, 1))

            q = c_T_i @ i_T_w @ m_j
            d_pi = np.squeeze(projectionJacobian(q.T))
            HM[4*i:(4*i+4), 3*j_idx:(3*j_idx+3)] = my_robot.Ks @ d_pi @ c_T_i @ i_T_w @ self.P.T
            i += 1

        #create HR matrix for 
        for j in range(len(update_idx)):
            j_idx = update_idx[j]

            mu_j = map_pts[3*j_idx:(3*j_idx+3),0].reshape(3,1)
            m_j = np.vstack((mu_j, 1))

            q = c_T_i @ i_T_w @ m_j
            d_pi = np.squeeze(projectionJacobian(q.T))
            HR[4*j:(4*j+4), :] = -my_robot.Ks @ d_pi @ c_T_i @ circle_dot(i_T_w @ m_j)

        H = np.hstack((HR, HM))
        KG = sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + IV)
        innovation = zt - pred_z[:, update_idx].flatten(order="F")

        KG_pose = KG[:6, :]
        KG_map = KG[6:, :]

        del_mu_pose = KG_pose @ innovation
        del_mu_map = KG_map @ innovation

        del_pose = np.squeeze(pr3_utils.axangle2pose(np.expand_dims(del_mu_pose, axis=0)))
        pose = pose @ del_pose
        map_pts = map_pts + del_mu_map.reshape(-1,1)

        sigma = (np.eye(3*n_landmarks + 6) - KG @ H) @ sigma

        return pose,  map_pts, sigma

    def init_landmarks(self, m_state, sigma, w_T_mloc, sub_features, init_list):
        for i in range(len(init_list)):
            idx = init_list[i]
            m_state[3*idx:(3*idx+3), 0] = w_T_mloc[:,idx]

        return m_state, sigma

def hat_operator(vec):
    mat = np.zeros((3,3))
    mat[0,1] = -vec[2]
    mat[1,0] = vec[2]

    mat[0,2] = vec[1]
    mat[2,0] = -vec[1]

    mat[1,2] = -vec[0]
    mat[2,1] = vec[0]
    return mat

def circle_dot(vec):
    mat = np.zeros((4,6))
    mat[:3,:3] = np.eye(3)
    mat[:3,3:] = -pr3_utils.axangle2skew(vec.T)
    return mat
