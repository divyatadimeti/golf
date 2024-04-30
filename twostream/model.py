import torch
import torch.nn as nn
from torch.autograd import Variable
from MobileNetV2 import MobileNetV2

class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, bidirectional=True, dropout=True, pose_dim=32):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.pose_dim = pose_dim

        # MobileNetV2 as the CNN for video frames
        net = MobileNetV2(width_mult=width_mult)
        state_dict_mobilenet = torch.load('mobilenet_v2.pth.tar')
        if pretrain:
            net.load_state_dict(state_dict_mobilenet)

        self.cnn = nn.Sequential(*list(net.children())[0][:19])

        # LSTM for the video frames
        self.rnn_video = nn.LSTM(int(1280 * width_mult if width_mult > 1.0 else 1280),
                                 self.lstm_hidden, self.lstm_layers,
                                 batch_first=True, bidirectional=bidirectional)

        # Simple feed-forward network for pose data
        self.pose_net = nn.Sequential(
            nn.Linear(pose_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # LSTM for the pose data
        self.rnn_pose = nn.LSTM(256, self.lstm_hidden, self.lstm_layers,
                                batch_first=True, bidirectional=bidirectional)

        # Linear layer to combine outputs
        if self.bidirectional:
            self.lin = nn.Linear(2 * self.lstm_hidden * 2, 9)  # Combine bidirectional outputs
        else:
            self.lin = nn.Linear(self.lstm_hidden * 2, 9)

        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        # Initialize hidden state for both LSTMs
        return (Variable(torch.zeros(2 * self.lstm_layers if self.bidirectional else self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
                Variable(torch.zeros(2 * self.lstm_layers if self.bidirectional else self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True))

    def forward(self, x, poses, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        self.hidden_video = self.init_hidden(batch_size)
        self.hidden_pose = self.init_hidden(batch_size)

        # CNN forward for video frames
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)
        if self.dropout:
            c_out = self.drop(c_out)

        # Unpack pose data from the dictionaries
        pose_data = []
        for video_id, joint_info in poses.items():
            for frame_data in joint_info:
                # Assuming the frame data is a dictionary with the joint data
                joint_data = list(frame_data.values())
                pose_data.extend(joint_data)
        pose_data = torch.tensor(pose_data, dtype=torch.float32).view(batch_size, timesteps, -1)

        # Pose network forward
        p_out = self.pose_net(pose_data)
        r_out_pose, self.hidden_pose = self.rnn_pose(p_out, self.hidden_pose)

        # LSTM forward for video frames
        r_out_video, self.hidden_video = self.rnn_video(c_out.view(batch_size, timesteps, -1), self.hidden_video)

        # Combine outputs
        combined = torch.cat((r_out_video, r_out_pose), dim=2)
        out = self.lin(combined.view(batch_size * timesteps, -1))

        return out


