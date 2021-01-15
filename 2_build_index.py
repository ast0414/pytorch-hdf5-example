import h5py
import os
import pickle

import numpy as np
from tqdm.auto import tqdm


H5_PATH = "/home/san37/Datasets/UMich/example.h5"
SAVE_DIR = "/home/san37/Datasets/UMich/"

WINDOW_SIZE = 256
STEP = WINDOW_SIZE // 2

TEST_SUBJECTS = {"Subject01"}
VAL_SUBJECTS = {"Subject02"}

train_idx, val_idx, test_idx = [], [], []

with h5py.File(H5_PATH, 'r') as hf:
    for subject_id in tqdm(hf.keys(), desc='Subject', leave=True):
        for protocol in tqdm(hf[subject_id].keys(), desc='Protocol', leave=False):

            phase = np.asarray(hf[subject_id][protocol]['Phase'])

            for i in tqdm(range(0, len(phase) - WINDOW_SIZE + 1, STEP), desc="Window", leave=False):

                act_label = phase[i:i + WINDOW_SIZE]
                label = np.unique(act_label).astype(np.int)

                # Ignore transitioning states for now
                if len(label) == 1:
                    if subject_id in TEST_SUBJECTS:
                        test_idx.append((subject_id, protocol, i, int(label[0])))
                    elif subject_id in VAL_SUBJECTS:
                        val_idx.append((subject_id, protocol, i, int(label[0])))
                    else:
                        train_idx.append((subject_id, protocol, i, int(label[0])))
                else:
                    continue

all_idx = {'train': train_idx, 'val': val_idx, 'test': test_idx}
with open(os.path.join(SAVE_DIR, "ex_idx_list.pkl"), 'wb') as f:
    pickle.dump(all_idx, f, pickle.HIGHEST_PROTOCOL)