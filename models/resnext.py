from __future__ import division
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import pdb

__all__ = ['ResNeXt', 'resnet50', 'resnet101']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400,
                 input_channels=3,
                 output_layers=[]):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            input_channels,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        
        #layer to output on forward pass
        self.output_layers = output_layers

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        #pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)

        x6 = x5.view(x5.size(0), -1)
        x7 = self.fc(x6)
        
        if len(self.output_layers) == 0:
            return x7
        else:
            out = []
            out.append(x7)
            for i in self.output_layers:
                if i == 'avgpool':
                    out.append(x6)
                if i == 'layer4':
                    out.append(x4)
                if i == 'layer3':
                    out.append(x3)
                if i=='dict':
                    out = {
                        'x': x,
                        'x1': x1,
                        'x2': x2,
                        'x3': x3,
                        'x4': x4,
                        'x5': x5,
                        'x7': x7,
                    }

        return out

    def freeze_batch_norm(self):
        for name,m in self.named_modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d): # PHIL: i Think we can write just  "if isinstance(m, nn._BatchNorm)
                m.eval() # use mean/variance from the training
                m.weight.requires_grad = False
                m.bias.requires_grad = False


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    print("Layers to finetune : ", ft_module_names)

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model

class new_fusion_model(nn.Module):
    """ attention """

    def __init__(self, feat_dim=400):
        super(new_fusion_model, self).__init__()
        self.feat_dim = feat_dim
        self.weight_layer = nn.Linear(feat_dim * 2, feat_dim*2*2)
        self.weight_layer2=nn.Linear(feat_dim*2*2,feat_dim * 2)
        self.fc = nn.Linear(feat_dim * 2, feat_dim)

        # self.fc = nn.Linear(feat_dim, feat_dim)
        # self.weight_layer.bias.data.zero_()
        # self.weight_layer.weight.data.zero_()
        # self.weight_layer2.bias.data.zero_()
        # self.weight_layer2.weight.data.zero_()
        # nn.init.xavier_uniform_(self.fc.weight)

        # nn.init.xavier_uniform_(self.q)
        # nn.init.xavier_uniform_(self.fc.weight)
        # nn.init.constant_(self.fc.bias, 0)
        # self.fc.bias.data.zero_()  # NAN init these paprameter as zeros
        # self.fc.weight.data.zero_()

    def forward(self, x_mar,x_flow):
        # Xs: batch*feature-length*seq                              [Batch,Dimension]
        x_mar=F.normalize(x_mar)
        x_flow=F.normalize(x_flow)
        X=torch.cat([x_mar,x_flow],-1)
        weights=F.softmax(self.weight_layer2(self.weight_layer(X)),dim=-1)
        r = torch.mul(X, weights)
        Y=torch.sum(r.view(-1,self.feat_dim,2),dim=-1)
        return weights.squeeze(),Y


class fusion_model(nn.Module):
    """ attention """

    def __init__(self, feat_dim=400):
        super(fusion_model, self).__init__()
        self.feat_dim = feat_dim
        self.weight_layer = nn.Linear(feat_dim * 2, int(feat_dim/2))
        self.weight_layer2=nn.Linear(int(feat_dim/2),2)
        self.fc = nn.Linear(feat_dim * 2, feat_dim)

        # self.fc = nn.Linear(feat_dim, feat_dim)
        # self.weight_layer.bias.data.zero_()
        # self.weight_layer.weight.data.zero_()
        # self.weight_layer2.bias.data.zero_()
        # self.weight_layer2.weight.data.zero_()
        # nn.init.xavier_uniform_(self.fc.weight)

        # nn.init.xavier_uniform_(self.q)
        # nn.init.xavier_uniform_(self.fc.weight)
        # nn.init.constant_(self.fc.bias, 0)
        # self.fc.bias.data.zero_()  # NAN init these paprameter as zeros
        # self.fc.weight.data.zero_()

    def forward(self, x_mar,x_flow):
        # Xs: batch*feature-length*seq                              [Batch,Dimension]
        x_mar=F.normalize(x_mar)
        x_flow=F.normalize(x_flow)
        X=torch.cat([x_mar,x_flow],-1)
        Xs= torch.stack([x_mar, x_flow], -1)
        weights=F.sigmoid(self.weight_layer2(self.weight_layer(X)))
        weights=weights.unsqueeze(1)
        r = torch.mul(Xs, weights)
        Y=torch.sum(r,dim=-1)
        return [weights.squeeze(),Y]

class Attention(nn.Module):
    """ attention """

    def __init__(self, feat_dim=128):
        super(Attention, self).__init__()
        self.feat_dim = feat_dim
        self.q_mar= nn.Parameter(torch.ones((1,1,feat_dim)) * 0.0, requires_grad=True)
        self.q_flow = nn.Parameter(torch.ones((1,1, feat_dim)) * 0.0, requires_grad=True)

        # self.q = nn.Parameter(torch.ones((1, 1, feat_dim)), requires_grad=True)

        self.fc = nn.Linear(feat_dim, feat_dim)
        self.tanh = nn.Tanh()

        # nn.init.xavier_uniform_(self.q)
        # nn.init.xavier_uniform_(self.fc.weight)
        # nn.init.constant_(self.fc.bias, 0)
        self.fc.bias.data.zero_()  # NAN init these paprameter as zeros
        self.fc.weight.data.zero_()

    def squash(self, x):
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        v = (x2 / (1.0 + x2)) * (x / x2.sqrt())
        return v

    def forward(self, x_mar,x_flow):
        # Xs: batch*feature-length*seq                              [Batch,Dimension]
        x_mar=F.normalize(x_mar)
        x_flow=F.normalize(x_flow)
        Xs=torch.stack([x_mar,x_flow],-1)
        N, C, K = Xs.shape  # N: batch C: channel(feature dimention) K: Frames  [3,128,20]
        score1 = torch.matmul(self.q_mar, x_mar.unsqueeze(-1))  # N*1*K ==[1,2,128] *[3,128,1]
        score2 = torch.matmul(self.q_flow, x_flow.unsqueeze(-1))  # N*1*K ==[1,2,128] *[3,128,20]
        score=torch.cat([score1, score2], -1)
        score = F.softmax(score, dim=-1)
        r = torch.mul(Xs, score)  # element-wise multiply  [3,128,2]
        r = torch.sum(r, dim=-1)  # N*C  [3,128]

        new_q = self.fc(r)  # N*C
        new_q = self.tanh(new_q)
        new_q = new_q.view(N, 1, C)

        new_score = torch.matmul(new_q, Xs)
        new_score = F.softmax(new_score, dim=-1)

        o = torch.mul(Xs, new_score)
        o = torch.sum(o, dim=-1)  # N*C

        # o = self.squash(o)

        return o
