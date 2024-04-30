import os.path as osp
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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
    def __init__(self, data_file, vid_dir, pose_data_file, seq_length, transform=None, train=True):
        self.df = pd.read_pickle(data_file)
        self.pose_df = pd.read_pickle(pose_data_file)  # Load pose data
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_info = self.df.iloc[idx]
        video_id = video_info['id']
        events = video_info['events']
        events -= events[0]  # Normalize frame numbers to start from 0

        # Load video frames
        images, labels = [], []
        cap = cv2.VideoCapture(osp.join(self.vid_dir, f"{video_id}.mp4"))
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        selected_frames = np.linspace(0, frame_count - 1, self.seq_length, dtype=int) if self.train else np.arange(frame_count)
        
        for frame_idx in selected_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, img = cap.read()
            if not ret:
                continue  # Skip if the frame was not captured successfully
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            images.append(img)
            label = np.where(events[1:-1] == frame_idx)[0][0] if frame_idx in events[1:-1] else 8
            labels.append(label)
        cap.release()
        
        print("got to here")
        # Retrieve the pose information for the entire video as one entry
        poses = self.pose_df.loc[self.pose_df['video_id'] == video_id, 'joint_info'].iloc[0]
        print("also got to here")
        sample = {'images': np.stack(images), 'poses': poses, 'labels': np.array(labels)}
        return sample

if __name__ == '__main__':
    transform = transforms.Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = GolfDB(data_file='/home/dt2760/golf/data/train_split_1.pkl',
                     vid_dir='/home/dt2760/golf/data/videos_160/',
                     pose_data_file = '/home/dt2760/golf/data/joint_info.pkl',
                     seq_length=64,
                     transform=transform,
                     train=False)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, drop_last=False)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        events = np.where(labels.squeeze() < 8)[0]
        print(f'{len(events)} events: {events}')
