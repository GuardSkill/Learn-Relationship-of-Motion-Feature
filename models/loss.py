import torch.nn as nn
import torch


class ModelGramLoss(nn.Module):
    def __init__(self):
        super(ModelGramLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, t, h, w = x.size()
        f = x.view(b, ch, t * w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (t * h * w * ch)

        return G

    def __call__(self, x_out, y_out):
        # Compute features
        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_out['x']), self.compute_gram(y_out['x']))
        style_loss += self.criterion(self.compute_gram(x_out['x1']), self.compute_gram(y_out['x1']))
        style_loss += self.criterion(self.compute_gram(x_out['x2']), self.compute_gram(y_out['x2']))
        style_loss += self.criterion(self.compute_gram(x_out['x3']), self.compute_gram(y_out['x3']))
        style_loss += self.criterion(self.compute_gram(x_out['x4']), self.compute_gram(y_out['x4']))
        style_loss += self.criterion(self.compute_gram(x_out['x5']), self.compute_gram(y_out['x5']))
        style_loss/=6
        return style_loss


class ModelL1Loss(nn.Module):
    def __init__(self):
        super(ModelL1Loss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def __call__(self, x_out, y_out):
        # Compute features
        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(x_out['x'], y_out['x'])
        style_loss += self.criterion(x_out['x1'], y_out['x1'])
        style_loss += self.criterion(x_out['x2'], y_out['x2'])
        style_loss += self.criterion(x_out['x3'], y_out['x3'])
        style_loss += self.criterion(x_out['x4'], y_out['x4'])
        style_loss += self.criterion(x_out['x5'], y_out['x5'])
        style_loss/=6
        return style_loss
