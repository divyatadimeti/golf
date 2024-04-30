from model import EventDetector
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds
import sys

def eval(model, split, seq_length, n_cpu, disp):
    # No need to adjust for DimUpperHalf since we are evaluating
    dataset = GolfDB(data_file=f'/home/dt2760/golf/data/val_split_{split}.pkl',
                     vid_dir='/home/dt2760/golf/data/videos_160/',
                     pose_data_file=f'/home/dt2760/golf/data/joint_info.pkl',  # Add your pose data file path here
                     seq_length=seq_length,
                     transform=transforms.Compose([
                         ToTensor(),
                         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ]),
                     train=False)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    correct = []

    for i, sample in enumerate(data_loader):
        images, poses, labels = sample['images'], sample['poses'], sample['labels']
        batch = 0
        probs = []
        while batch * seq_length < images.shape[1]:
            end = min((batch + 1) * seq_length, images.shape[1])
            image_batch = images[:, batch * seq_length:end, :, :, :]
            pose_batch = poses[batch * seq_length:end]
            logits = model(image_batch.cuda(), pose_batch.cuda())
            probs_batch = F.softmax(logits, dim=1).detach().cpu().numpy()
            probs.append(probs_batch)
            batch += 1
        probs = np.concatenate(probs, axis=0)
        _, _, _, _, c = correct_preds(probs, labels.squeeze())
        if disp:
            print(f'Batch {i}: Correctly detected events: {c}')
        correct.append(c)

    PCE = np.mean(correct)
    return PCE

if __name__ == '__main__':
    split = 1
    seq_length = 64
    n_cpu = 6

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False,
                          pose_dim=32)  # Match this to your pose feature dimension

    save_dict = torch.load('/home/dt2760/golf/models/swingnet_with_pose.pth.tar')
    state_dict = save_dict['model_state_dict']

    # Adjust the keys in the state dict
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    
    with open('evaluation_results_with_pose.txt', 'w') as file:
        original_stdout = sys.stdout
        sys.stdout = file

        PCE = eval(model, split, seq_length, n_cpu, True)
        print(f'Average PCE: {PCE:.2f}%')

        sys.stdout = original_stdout

    print(f'Evaluation complete. Results saved to "evaluation_results_with_pose.txt".')

