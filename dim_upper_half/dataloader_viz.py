import os
import os.path as osp
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

class DimUpperHalf(object):
    def __init__(self):
        self.ensure_dir_exists('frames')

    def ensure_dir_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
    """Dim the upper half of each frame in the video."""
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        c, h, w = images.shape[1], images.shape[2], images.shape[3]
        
        # Save original images for inspection before dimming
        for i, img in enumerate(images):
            img_np = img.permute(1, 2, 0).mul(255).byte().cpu().numpy()  # Convert to HWC format, 8-bit and NumPy
            cv2.imwrite(f'frames/original_frame_{i}.png', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        # Apply dimming to the upper half of each frame
        images[:, :, :h//2, :] *= 0.5

        # Save dimmed images for inspection after dimming
        for i, img in enumerate(images):
            img_np = img.permute(1, 2, 0).mul(255).byte().cpu().numpy()  # Convert to HWC format, 8-bit and NumPy
            cv2.imwrite(f'frames/dimmed_frame_{i}.png', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        return {'images': images, 'labels': labels}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = images.transpose((0, 3, 1, 2))  # Convert shape from (T, H, W, C) to (T, C, H, W)
        return {'images': torch.from_numpy(images).float().div(255.),
                'labels': torch.from_numpy(labels).long()}

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'labels': labels}

class GolfDB(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, transform=None, train=True):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df.loc[idx]  # annotation info
        events = a['events']
        events -= events[0]  # adjust so frames start at 0

        images, labels = [], []
        cap = cv2.VideoCapture(osp.join(self.vid_dir, f'{a["id"]}.mp4'))

        if self.train:
            start_frame = np.random.randint(events[-1] + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            pos = start_frame
            while len(images) < self.seq_length:
                ret, img = cap.read()
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(np.where(events[1:-1] == pos)[0][0] if pos in events[1:-1] else 8)
                    pos += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pos = 0
        else:
            for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, img = cap.read()
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(np.where(events[1:-1] == pos)[0][0] if pos in events[1:-1] else 8)
        cap.release()

        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == '__main__':
    transform = transforms.Compose([
        ToTensor(),
        DimUpperHalf(),  # Apply the new DimUpperHalf transformation after converting to tensor
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = GolfDB(data_file='data/train_split_1.pkl',
                     vid_dir='data/videos_160/',
                     seq_length=64,
                     transform=transform,
                     train=False)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, drop_last=False)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        events = np.where(labels.squeeze() < 8)[0]
        print(f'{len(events)} events: {events}')
