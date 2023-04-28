# robot.py
import numpy as np
import pr3_utils

# 2D Robot
class robot:
    def __init__(self, K=None, b=None, iTc=None) -> None:
        self.pose = np.zeros((3,1))
        self.iTc = iTc
        self.K = K
        self.baseline = b
        self.Ks = np.zeros((4,4))
        self.Ks[:3,:3] = K
        self.Ks[2,0] = K[0,0]
        self.Ks[2,2] = K[0,2]
        self.Ks[3,:3] = self.Ks[1,:3]
        self.Ks[2,3] = -K[0,0]*b
        self.cTi = pr3_utils.inversePose(self.iTc)

    def predict_pose(self, dt, T, velocity, angular_vel):
        twist = np.hstack((velocity,angular_vel))
        twist = dt * twist
        twist = np.expand_dims(twist, axis=0)
        twist_pose = np.squeeze(pr3_utils.axangle2pose(twist))
        T_new = T @ twist_pose
        return T_new

    def set_pose(self, pose):
        self.pose = pose
        return pose

    def get_pos_from_stereo(self, stereo_meas):

        # filt detected measurements
        d = np.ones_like(stereo_meas[0,:])        

        valid_idx = stereo_meas[0,:] > 0
        d[valid_idx] = stereo_meas[0, valid_idx] - stereo_meas[2, valid_idx]

        z = self.K[0,0] * self.baseline/d
        x = z * (stereo_meas[0,:]-self.K[0,2])/self.K[0,0]
        y = z * (stereo_meas[1,:]-self.K[1,2])/self.K[1,1]
        camPOS = np.vstack((x,y,z,np.ones_like(z)))
        imuPOS = self.iTc @ camPOS

        return imuPOS
    
