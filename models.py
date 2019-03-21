import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolution(nn.Module):
    """
    convolutional layers
    """

    def __init__(self):
        super(Convolution, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(24)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        return x


class ObjectPair(nn.Module):
    """
    make object pair
    """

    def __init__(self):
        super().__init__()
        coord_tensor = torch.zeros((1, 25, 2), dtype=torch.float)
        for i in range(25):
            coord_tensor[:, i, :] = torch.tensor([(i // 5 - 2) / 2., (i % 5 - 2) / 2.], dtype=torch.float)
        self.register_buffer('coord_tensor', coord_tensor)

    def forward(self, x, qst):
        batch_size, n_channels, feature_dim, _ = x.shape
        coord_tensor = self.coord_tensor.repeat(batch_size, 1, 1)  # copy coordinate
        x = x.view(batch_size, n_channels, feature_dim ** 2).permute(0, 2, 1)
        x = torch.cat([x, coord_tensor], 2)
        qst = qst.reshape(batch_size, 1, 1, 11)
        qst = qst.repeat(1, feature_dim ** 2, feature_dim ** 2, 1)
        x_i = torch.unsqueeze(x, 1)
        x_i = x_i.repeat(1, feature_dim ** 2, 1, 1)
        x_j = torch.unsqueeze(x, 2)
        x_j = x_j.repeat(1, 1, feature_dim ** 2, 1)
        x = torch.cat([x_i, x_j, qst], 3)
        x = x.view(-1, feature_dim ** 4, (n_channels + 2) * 2 + 11)
        return x


class GTheta(nn.Module):
    """
    the g_\theta function
    """

    def __init__(self, ):
        super().__init__()
        self.g1 = nn.Linear(63, 256)
        self.g2 = nn.Linear(256, 256)
        self.g3 = nn.Linear(256, 256)
        self.g4 = nn.Linear(256, 256)

    def forward(self, x):
        x = F.relu(self.g1(x))
        x = F.relu(self.g2(x))
        x = F.relu(self.g3(x))
        x = F.relu(self.g4(x)).sum(1)
        return x


class FPhi(nn.Module):
    """
    the f_\phi function
    """

    def __init__(self):
        super(FPhi, self).__init__()
        self.f_fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.f_fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class RelationalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Convolution()
        self.obj_pair = ObjectPair()
        self.g_theta = GTheta()
        self.f_phi = FPhi()

    def forward(self, x, qst):
        x = self.conv(x)
        x = self.obj_pair(x, qst)
        x = self.g_theta(x)
        x = self.f_phi(x)
        return x
