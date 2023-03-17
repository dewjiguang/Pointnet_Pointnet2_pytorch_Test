import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
import torch.nn as nn
# import math
# try:
#     from torch.hub import load_state_dict_from_url
# except ImportError:
#     from torch.utils.model_zoo import load_url as load_state_dict_from_url
# import torch
#
# # 通道注意力
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         #自适应池化成1*1
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         # //除后取整
#         self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
# # 空间注意力
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         # 就一个卷积核，所以输出通道为1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         #dim,为1时，求得是行的平均值；为0时，求得是列的平均值。
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         # cat拼接二维数组，dim=0，拼接行，dim=1，拼接列（（3，3）+（3，3）=（3，6））
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)


class get_model(nn.Module):
    def __init__(self, num_class):
        super(get_model, self).__init__()
        self.k = num_class
        #k：k分类
        # 猜测PointNetEncoder为pointNet结构图上边那一大部分的升维池化提取特征部分的集合，他给封装了。
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=9)
        #下面这一部分是对每个点的1099个特征进行一维卷积，最终直接卷成k分类个，看每个的概率那个大就分到哪一类
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        # dilation=2不增加计算量的前提下，增大感受野 若卷积核改为5的话，为保证尺寸不变，加上padding=2
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        # BatchNorm1d  用于维持归一化的一个方法，具体原理不用知道，反正就是方便训练的
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        # self.ca = ChannelAttention(self.k)
        # self.sa = SpatialAttention()

    def forward(self, x):
        batchsize = x.size()[0] #维度？
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        # transpose:交换两个矩阵的两个维度，例如：
        # >> > x = torch.randn(2, 3)
        # tensor([[1.0028, -0.9893, 0.5809],
        #         [-0.1669, 0.7299, 0.4942]])
        # >> > torch.transpose(x, 0, 1)
        # tensor([[1.0028, -0.1669],
        #         [-0.9893, 0.7299],
        #         [0.5809, 0.4942]])
        # contiguous为将处理后的tensor变为内存连续
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        # x = self.ca(x) * x
        # x = self.sa(x) * x
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        loss = F.nll_loss(pred, target, weight = weight)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


if __name__ == '__main__':
    model = get_model(5) # 13->5
    xyz = torch.rand(12, 3, 2048)
    (model(xyz))