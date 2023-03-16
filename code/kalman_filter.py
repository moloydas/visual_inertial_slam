#kalman_filter.py
import numpy as np
from pr3_utils import *
from robot import *

# Kalman Filter
class kalman_filter:
    def __init__(self, vx=0.1, vz=5) -> None:
        self.vz = vz
        self.vx = vx
        self.W = np.eye(6)
        self.W[:3,:3] *= .3 #vx**2
        self.W[3:,3:] *= .3 #vx**2

    def prediction(self, robot, dt, mu, velocity, angular_vel, sigma):
        mu_t = robot.predict_pose(dt, mu, velocity, angular_vel)
        twist = np.hstack((velocity,angular_vel))
        twist = np.expand_dims(twist, axis=0)
        # twist_pose = -dt * pr3_utils.axangle2pose(twist)
        # adjoint = np.squeeze(pr3_utils.pose2adpose(np.expand_dims(twist_pose, axis=0)))
        adjoint = np.squeeze(pr3_utils.axangle2adtwist(-dt*twist))
        sigma_t1 = adjoint @ sigma @ adjoint.T + self.W
        return mu_t, sigma_t1

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

        KG = sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + IV)
        zt = measurements.flatten(order="F")
        q = c_T_i @ i_T_w @ norm_mu_t
        pred_z = my_robot.Ks @ projection(q.T).T
        innovation = zt - pred_z[:, update_idx].flatten(order="F")
        del_mu = KG @ innovation
        mu_t1 = mu_t + del_mu.reshape(-1,1)
        sigma_t1 = (np.eye(3*n_landmarks) - KG @ H) @ sigma
        return mu_t1, sigma_t1

    def update_pose(self, my_robot, mu, pose_sigma, i_T_c, map_pts, measurements, update_idx):
        P = np.zeros((3,4))
        P[:,:3] = np.eye(3)
        nt = measurements.shape[1]
        H = np.zeros((4*nt, 6))

        c_T_i = inversePose(i_T_c)
        IV = np.eye(4*nt) * self.vz

        norm_map = map_pts.T.reshape(-1,3).T
        norm_map = np.vstack((norm_map, np.ones((1, norm_map.shape[1]))))
        q = c_T_i @ inversePose(mu) @ norm_map
        pred_z = my_robot.Ks @ projection(q.T).T

        i=0
        for j in range(len(update_idx)):
            j_idx = update_idx[j]

            mu_j = map_pts[3*j_idx:(3*j_idx+3),0].reshape(3,1)
            m_j = np.vstack((mu_j, 1))

            q = c_T_i @ inversePose(mu) @ m_j
            d_pi = np.squeeze(projectionJacobian(q.T))
            H[4*j:(4*j+4), :] = -my_robot.Ks @ d_pi @ c_T_i @ circle_dot(pr3_utils.inversePose(mu) @ m_j)
            i += 1

        KG = pose_sigma @ H.T @ np.linalg.inv(H @ pose_sigma @ H.T + IV)
        zt = measurements.flatten(order="F")

        innovation = zt - pred_z[:, update_idx].flatten(order="F")
        del_mu = KG @ innovation
        mu_t1 = mu @ pr3_utils.axangle2pose(np.expand_dims(del_mu, axis=0))
        sigma_t1 = (np.eye(6) - KG @ H) @ pose_sigma
        return mu_t1, sigma_t1


    def init_landmarks(self, m_state, sigma, w_T_mloc, sub_features, init_list):
        # itr = 0
        # for i in range(len(init_list)):
        #     if sub_features[0,i] > 0 and m_state[3*i,0] == -5000:
        #         m_state[3*i:(3*i+3), 0] = w_T_mloc[:,itr]
        #         sigma[3*i:3*i+3,3*i:3*i+3]  = np.eye(3) * (self.vz)**2
        #         itr+=1
        #     elif sub_features[0,i] > 0 and m_state[3*i,0] != -5000:
        #         itr+=1

        for i in range(len(init_list)):
            idx = init_list[i]
            m_state[3*idx:(3*idx+3), 0] = w_T_mloc[:,i]
            sigma[3*idx:3*idx+3,3*idx:3*idx+3]  = np.eye(3) * (self.vx)**2

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
    mat[:3,3:] = -hat_operator(vec)
    return mat
