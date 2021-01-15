import argparse
import h5py
import pickle
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import AccelH5Dataset
from transforms import ChannelStandardization, MapLabel


def main(args):

    # Some settings
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cudnn.benchmark = True if args.cuda else False
    device = torch.device("cuda" if args.cuda else "cpu")

    # Load pre-built index
    with open(args.index, 'rb') as f:
        idx_list = pickle.load(f)

    idx_train = idx_list['train']
    idx_val = idx_list["val"]
    idx_test = idx_list["val"]

    # You may want to apply some sample-wise transforms
    transform = transforms.Compose(
        [
            ChannelStandardization(),
        ]
    )

    target_transform = MapLabel()

    hf = h5py.File(args.h5, mode='r', driver='core')
    train_dataset = AccelH5Dataset(idx_train, hf, len_frame=args.len_frame, transform=transform, target_transform=target_transform)
    val_dataset = AccelH5Dataset(idx_val, hf, len_frame=args.len_frame, transform=transform, target_transform=target_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=int(args.num_workers), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=int(args.num_workers), drop_last=False)

    # Simple model - we do not expect this model performs well of course
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=3*args.len_frame, out_features=2)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(10):

        # Train
        model.train()
        for i, (data, target) in enumerate(train_loader):

            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print("Epoch {}, Step {}, Loss {}".format(epoch, i, loss.item()))

        # Validation
        model.eval()
        # skip for now


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--len-frame', type=int, default=256, metavar='N', help='length of signal (default: 256)')
    parser.add_argument('--index', default='/home/san37/Datasets/UMich/ex_idx_list.pkl', type=str, metavar='S', help='index list')
    parser.add_argument('--h5', type=str, default='/home/san37/Datasets/UMich/example.h5', help='path to the dataset HDF5 file')

    parser.add_argument('--num-workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    args = parser.parse_args()

    main(args)
