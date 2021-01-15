import numpy as np
from torch.utils.data import Dataset


class AccelH5Dataset(Dataset):
    def __init__(self, list_index, hf, len_frame, transform=None, target_transform=None):
        """
        :param list_index (list): List containing index information (subject id, frame id, label)
        :param hf (HDF5 file: H5 dataset object already opened.
        :param transform (callable, optional): Optional transform to be applied on a sample.
        :param target_transform (callable, optional): Optional transform to be applied on a target.
        """
        self.hf = hf
        self.len_frame = len_frame
        self.list_index = list_index
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        subject_id, protocol, frame_idx, target = self.list_index[index]

        h5_key = "{}/{}/Accel".format(subject_id, protocol)
        sample = np.asarray(self.hf[h5_key][:, frame_idx:frame_idx + self.len_frame])

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.list_index)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
