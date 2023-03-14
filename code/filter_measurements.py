#filter_measurements.py
import numpy as np
from pr3_utils import *

# returns indexs of the subsampled measurements
# reduction of features based on disparity
def disparity_filter_measurements(all_measurements, n=5000, d_thres=1):
    all_measurements = random_subsample_measurements(all_measurements, n)
    d = all_measurements[0, :, :] - all_measurements[2, :, :]
    valid_d_idx = np.logical_or(d > d_thres, d == 0)
    valid_d_idx = np.all(valid_d_idx, axis=1)
    
    return all_measurements[:,valid_d_idx,:]

def random_subsample_measurements(all_measurements, n=5000):
    np.random.seed(0)
    n_landmark = all_measurements.shape[1]
    samples = np.random.choice(n_landmark, size=n, replace=False)
    return all_measurements[:, samples, :]

if __name__ == '__main__':
    # Load the measurements
    filename = "../data/10.npz"
    t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
    # sub_features = random_subsample_measurements(features, 5000)

    print(features.shape)
    sub_measurements = disparity_filter_measurements(features)
    print(sub_measurements.shape)




