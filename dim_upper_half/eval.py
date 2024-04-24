from model import EventDetector
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize, DimUpperHalf
import torch.nn.functional as F
import numpy as np
from util import correct_preds

def eval(model, split, seq_length, n_cpu, disp):
    # Adjust the transform sequence to include the DimUpperHalf
    dataset = GolfDB(data_file=f'data/val_split_{split}.pkl',
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([
                         ToTensor(),
                         DimUpperHalf(),  # Dimming the upper half of the video frames
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
        images, labels = sample['images'], sample['labels']
        # Evaluate the sample in 'seq_length' batches due to GPU memory constraints
        batch = 0
        probs = []
        while batch * seq_length < images.shape[1]:
            end = min((batch + 1) * seq_length, images.shape[1])
            image_batch = images[:, batch * seq_length:end, :, :, :]
            logits = model(image_batch.cuda())
            probs_batch = F.softmax(logits, dim=1).detach().cpu().numpy()  # Detach before converting to numpy
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
                          dropout=False)

    save_dict = torch.load('models/swingnet_dimmed.pth.tar')
    state_dict = save_dict['model_state_dict']

    # Adjust the keys in the state dict
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    
    # Redirecting print outputs to a file
    with open('evaluation_results_original_dimmed.txt', 'w') as file:
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = file  # Change the standard output to the file we created.

        PCE = eval(model, split, seq_length, n_cpu, True)
        print(f'Average PCE: {PCE:.2f}%')

        sys.stdout = original_stdout  # Reset the standard output to its original value

    print(f'Evaluation complete. Results saved to "evaluation_results_dimmed.txt".')

