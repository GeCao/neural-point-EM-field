import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List


# https://github.com/fxia22/pointnet.pytorch
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """See Input Transform (blank block) Figure 2 from: https://arxiv.org/pdf/1612.00593.pdf
        Args:
            x (torch.Tensor): [B, dim, num_pts]

        Returns:
            torch.Tensor: Learned Transform matrix [-1, dim, dim]

        - Very first three conv -> MLP(64, 128, 1024)
        - torch.max(x, dim=2) -> max
        - Last two full connect layer -> MLP(512, 256, dim**2)
        - finally add an identity matrix.
        """
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, num_pts]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, num_pts]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 1024, num_pts]
        x = torch.max(x, 2, keepdim=True)[0]  # [B, 1024, 1]
        x = x.view(-1, 1024)  # [B, 1024]

        if batchsize == 1:
            x = F.relu(self.fc1(x))  # [B, 512]
            x = F.relu(self.fc2(x))  # [B, 256]
        else:
            x = F.relu(self.bn4(self.fc1(x)))  # [B, 512]
            x = F.relu(self.bn5(self.fc2(x)))  # [B, 256]
        x = self.fc3(x)  # [B, dim**2]

        iden = (
            Variable(
                torch.from_numpy(
                    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
                )
            )
            .view(1, 9)
            .repeat(batchsize, 1)
        ).to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)  # [B, dim, dim]
        return x


class STNkd(nn.Module):
    def __init__(self, k: int = 64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """See Feature Transform (blank block) Figure 2 from: https://arxiv.org/pdf/1612.00593.pdf
        Args:
            x (torch.Tensor): [B, k, num_pts]

        Returns:
            torch.Tensor: Learned Transform matrix [B, k, k]

        - Very first three conv -> MLP(64, 128, 1024)
        - torch.max(x, dim=2) -> max
        - Last two full connect layer -> MLP(512, 256, k2)
        - finally add an identity matrix.
        """
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if batchsize == 1:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        else:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        ).to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat: bool = True, feature_transform: bool = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """See Blue & Blank block from Figure 2 from: https://arxiv.org/pdf/1612.00593.pdf
        Args:
            x (torch.Tensor): [B, 3, num_pts]

        Returns:
            torch.Tensor: Learned Point feature    [B, 1024]         / [B, 1088, num_pts]
            torch.Tensor: Input Transform Matrix   [B, 3, 3]         / [B, 3, 3]
            torch.Tensor: Feature Transform Matrix [B, 64, 64](None) / [B, 64, 64](None)

        - Very first STN3d -> input transform part
        - Following STN64d -> MLP(64, 64)
        - If branch:
            - True -> feature transform part
            - False -> Skip feature transform part
        - (64, 128, 1024) -> MLP: 64 --conv--> 128 --conv--> 1024
        - Max pool -> Global Feature
        - If branch:
            - True ->
            - False -> Prepare for segmentation Network
        """
        n_pts = x.size()[2]
        # input transform
        trans = self.stn(x)  # [B, 3, 3]
        x = x.transpose(2, 1)  # [B, num_pts, 3]
        x = torch.bmm(x, trans)  # [B, num_pts, 3]
        x = x.transpose(2, 1)  # [B, 3, num_pts]
        # (64,64)
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, num_pts]

        if self.feature_transform:
            # feature transform
            trans_feat = self.fstn(x)  # [B, 64, 64]
            x = x.transpose(2, 1)  # [B, num_pts, 64]
            x = torch.bmm(x, trans_feat)  # [B, num_pts, 64]
            x = x.transpose(2, 1)  # [B, 64, num_pts]
        else:
            trans_feat = None

        # (64, 128, 1024)
        pointfeat = x  # [B, 64, num_pts]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, num_pts]
        x = self.bn3(self.conv3(x))  # [B, 1024, num_pts]

        # Max pool  -> global feature
        x = torch.max(x, 2, keepdim=True)[0]  # [B, 1024, 1]
        x = x.view(-1, 1024)  # [B, 1024]
        if self.global_feat:
            # Returns only the global features
            return x, trans, trans_feat  # [B, 1024], [B, 3, 3], [B, 64, 64](or None)
        else:
            # Concatenates local and global features
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)  # [B, 1024, num_pts]
            # [B, 1088, num_pts], [B, 3, 3], [B, 64, 64](or None)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k=2, feature_transform: bool = False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """See Yellow block from Figure 2 from: https://arxiv.org/pdf/1612.00593.pdf
        Args:
            x (torch.Tensor): [B, 3, num_pts]

        Returns:
            torch.Tensor: Classification Score [B, num_pts, k]
            torch.Tensor: Input Transform Matrix   [B, 3, 3]         / [B, 3, 3]
            torch.Tensor: Feature Transform Matrix [B, 64, 64](None) / [B, 64, 64](None)

        - Very first Feat -> Get your Feature (Blue block from fig 2)
        - Following MLP -> MLP(512, 256, 128)
        - Log SoftMax -> result
        """
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # [B, 1088, num_pts], [B, 3, 3], [B, 64, 64]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)  # [B, k, num_pts]
        x = x.transpose(2, 1).contiguous()  # [B, num_pts, k]
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)  # [B, num_pts, k]
        return x, trans, trans_feat


class PointNetLightFieldEncoder(nn.Module):
    def __init__(
        self,
        k=2,
        feature_transform: bool = False,
        points_only: bool = False,
        original: bool = False,
    ):
        super(PointNetLightFieldEncoder, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.points_only = points_only
        self.original = original
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        if not points_only and not original:
            # Global Features only (Blue block from fig 2, https://arxiv.org/pdf/1612.00593.pdf)
            self.conv1 = torch.nn.Conv1d(1024, 256, 1)
            self.conv2 = torch.nn.Conv1d(256, 64, 1)
            self.conv3 = torch.nn.Conv1d(128, self.k, 1)

            self.bn1 = nn.BatchNorm1d(256)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(self.k)
        elif original:
            # Local & Global Features (Yellow block from fig 2, https://arxiv.org/pdf/1612.00593.pdf)
            self.conv1 = torch.nn.Conv1d(1088, 512, 1)
            self.conv2 = torch.nn.Conv1d(512, 256, 1)
            self.conv3 = torch.nn.Conv1d(256, 128, 1)
            self.conv4 = torch.nn.Conv1d(128, self.k, 1)
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(128)
        else:
            self.conv4 = torch.nn.Conv1d(64, self.k, 1)
            self.bn4 = nn.BatchNorm1d(self.k)

    def forward(self, x, **kwargs):
        """See Blue & Yellow block from Figure 2 from: https://arxiv.org/pdf/1612.00593.pdf
        Args:
            x (torch.Tensor): [B, 3, num_pts]

        Returns:
            torch.Tensor: Classification Score [B, k, num_pts]
            torch.Tensor: Input Transform Matrix   [B, 3, 3]         / [B, 3, 3]
            torch.Tensor: Feature Transform Matrix [B, 64, 64](None) / [B, 64, 64](None)

        - Very first Feat -> Get your Feature (Blue block from fig 2)
        - Following MLP -> MLP(256, 64, k) / MLP(512, 256, 128, k)
        - # Log SoftMax -> result
        """
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        point_feat = x[:, 1024:, :]
        glob_feat = x[:, :1024, :]

        if not self.points_only and not self.original:
            x = F.relu(self.bn1(self.conv1(glob_feat)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = torch.cat([x, point_feat], dim=1)
            x = F.relu(self.bn3(self.conv3(x)))
        elif self.original:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.conv4(x)
        else:
            x = F.relu(self.bn4(self.conv4(point_feat)))

        x = x.transpose(2, 1).contiguous()
        # x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat
