import glob
import h5py
import os

import numpy as np
from scipy.io import loadmat
from tqdm.auto import tqdm


DATA_DIR = "/home/san37/Datasets/UMich/"
SAVE_DIR = "/home/san37/Datasets/UMich/"
SAVE_H5_NAME = "example.h5"


def preprocess(signal):
    # DO SOMETHING
    return signal


record_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))

# Rows of the data
# 0: Time(s)
# 1: Activity Code
# ...
# 11-13: Chest Acceleration (m/s^2) x, y, z
# 14-16: Chest Angular Velocity (rad/s) x, y, z
# 17-19: Chest Magnetic Field (uT) x, y, z
# ...
rows = [1, 11, 12, 13]

with h5py.File(os.path.join(SAVE_DIR, SAVE_H5_NAME), 'w', libver='latest') as hf_write:

    for file in tqdm(record_files, desc="File"):
        subject_id = os.path.basename(file).split('.')[0]

        # Depending on the version of the target .MAT files (I guess?)
        # We may use either scipy.io.loadmat or h5py.File
        # Here, scipy.io.loadmat does not seem to work for these MAT files

        with h5py.File(file, 'r') as hf_read:
            for protocol in list(hf_read[subject_id].keys()):

                # Activity codes and Accelerometer (of IMU) located at Chest
                record = np.asarray(hf_read[subject_id][protocol]['APDM_Accel']['Data'][rows, :])
                phase = record[0, :].astype(np.int)
                accel = record[1:, :].astype(np.float32)

                # TODO: You may want to do some pre-processing of the entire signal here
                processed_accel = preprocess(accel)

                h5_key = "{}/{}".format(subject_id, protocol)
                tqdm.write(h5_key)

                hf_write.create_dataset(h5_key + '/Phase', shape=phase.shape, dtype=np.int, data=phase)
                hf_write.create_dataset(h5_key + '/Accel', shape=accel.shape, dtype=np.float32, data=accel)
