from dataloader import GolfDB, Normalize, ToTensor, DimUpperHalf
from model import EventDetector
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os

if __name__ == '__main__':

    # training configuration
    split = 1
    iterations = 10000
    n_cpu = 6
    seq_length = 64
    bs = 4  # batch size
    k = 10  # frozen layers

    # Initialize the model
    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    freeze_layers(k, model)
    model.train()
    model.cuda()

    # Prepare dataset
    dataset = GolfDB(data_file=f'data/train_split_{split}.pkl',
                     vid_dir='data/videos_160',
                     seq_length=seq_length,
                     transform=transforms.Compose([
                         ToTensor(),
                         DimUpperHalf(),
                         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ]),
                     train=True)

    data_loader = DataLoader(dataset,
                             batch_size=bs,
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True)

    # Weighting for class imbalance
    weights = torch.FloatTensor([1/8] * 8 + [1/35]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    losses = AverageMeter()

    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0
    # Training loop
    while i < iterations:
        for sample in data_loader:
            images, labels = sample['images'].cuda(), sample['labels'].cuda()
            logits = model(images)
            labels = labels.view(-1)  # Flatten labels to match output logits dimensions
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), images.size(0))

            print(f'Iteration: {i}\tLoss: {losses.val:.4f} ({losses.avg:.4f})')

            i += 1
            if i >= iterations:
                break

    # Save the final model weights
    torch.save({'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict()}, f'models/swingnet_dimmed_lower.pth.tar')

