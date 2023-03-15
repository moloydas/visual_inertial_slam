#kalman_filter.py
import numpy as np
from pr3_utils import *
from robot import *

# Kalman Filter
class kalman_filter:
    def __init__(self, dx=5000, vx=0.1, dz=5000, vz=5) -> None:
        self.W = np.eye(dx)*(vx)**2
        self.V = np.eye(dz)*(vz)**2
        self.vz = vz

    def prediction(self, mu_t, sigma):
        sigma_t1 = sigma + self.W
        return mu_t, sigma_t1

    def update(self, my_robot, sigma, i_T_c, w_T_i, mu_t, measurements, n_landmarks, update_idx):
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
        for j in range(n_landmarks):
            mu_j = mu_t[3*j:(3*j+3),0].reshape(3,1)
            if update_idx[j]:
                m_j = np.vstack((mu_j, 1))
                q = c_T_i @ i_T_w @ m_j
                d_pi = projectionJacobian(q.T)
                H[4*i:(4*i+4), 3*j:(3*j+3)] = my_robot.Ks @ d_pi @ c_T_i @ i_T_w @ P.T
                i += 1

        KG = sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + IV)
        zt = measurements.flatten(order="F")
        pred_z = my_robot.Ks @ projection(c_T_i @ i_T_w @ norm_mu_t)
        innovation = zt - pred_z[:, update_idx].flatten(order="F")
        del_mu = KG @ innovation
        mu_t1 = mu_t + del_mu.reshape(-1,1)
        sigma_t1 = (np.eye(3*n_landmarks) - KG @ H) @ sigma
        return mu_t1, sigma_t1

def init_landmarks(m_state, sigma, w_T_mloc, sub_features):
    itr = 0
    n_landmarks = sub_features.shape[1]
    for i in range(n_landmarks):
        if sub_features[0,i] > 0 and m_state[3*i,0] == -5000:
            m_state[3*i:(3*i+3), 0] = w_T_mloc[:,itr]
            sigma[3*i:3*i+3,3*i:3*i+3]  = np.eye(3)
            itr+=1
        elif sub_features[0,i] > 0 and m_state[3*i,0] != -5000:
            itr+=1

    return m_state, sigma


