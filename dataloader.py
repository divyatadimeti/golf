import os.path as osp
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class GolfDB(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, transform=None, train=True):
        self.df = pd.read_pickle(data_file)  # Load data frame with annotations
        self.vid_dir = vid_dir  # Directory with videos
        self.seq_length = seq_length  # Length of sequence to sample
        self.transform = transform  # Transformations to apply
        self.train = train  # Training mode toggle

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df.loc[idx]  # Simplified access
        events = a['events']
        events -= events[0]  # Normalize frame numbers

        images, labels = [], []
        cap = cv2.VideoCapture(osp.join(self.vid_dir, f"{a['id']}.mp4"))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.train:
            start_frame = np.random.randint(frame_count - self.seq_length + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            start_frame = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while len(images) < self.seq_length and cap.isOpened():
            ret, img = cap.read()
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.ndim == 2:
                    img = img[:, :, np.newaxis]  # Add channel dimension if missing
                images.append(img)
                pos = start_frame + len(images) - 1  # Current frame position
                label = np.where(events[1:-1] == pos, np.arange(len(events[1:-1])), 8)
                labels.append(label[0] if np.any(label < 8) else 8)
            else:
                break

        cap.release()

        images = np.array(images, dtype=np.uint8)
        labels = np.array(labels, dtype=np.int64)

        sample = {'images': images, 'labels': labels}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors and transpose the dimensions."""
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        # Transpose axis from (N, H, W, C) to (N, C, H, W)
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images).float() / 255,
                'labels': torch.from_numpy(labels)}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = (images - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        return {'images': images, 'labels': labels}


if __name__ == '__main__':

    transform_pipeline = transforms.Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = GolfDB(data_file='data/train_split_1.pkl',
                     vid_dir='data/videos_160/',
                     seq_length=64,
                     transform=transform_pipeline,
                     train=True)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        print(f"Batch {i}: Images shape {images.shape}, Labels shape {labels.shape}")
