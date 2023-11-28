import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet101
from utils import reparameterize
import torch.nn.utils.weight_norm as weightNorm
import numpy as np

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        try:
            nn.init.zeros_(m.bias)
        except:
            pass

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t) 
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, config):
        super(ConditionalModel, self).__init__()
        n_steps = config.diffusion.timesteps + 1
        data_dim = config.model.data_dim
        y_dim = config.data.num_classes
        feature_dim = config.model.feature_dim
        hidden_dim = config.model.hidden_dim
        # encoder for x (z)
        self.encoder_x = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, feature_dim)
        )
        # batch norm layer
        self.norm = nn.BatchNorm1d(feature_dim)

        # Unet
        self.lin1 = ConditionalLinear(y_dim * 2, feature_dim, n_steps)
        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, y_dim)

    def forward(self, x, y, t, yhat):
        x = self.encoder_x(x)
        x = self.norm(x)
        y = torch.cat([y, yhat], dim=-1)
        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        y = x * y
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        return self.lin4(y)


class ResNetBackbone(nn.Module):
  def __init__(self, resnet_name):
    super(ResNetBackbone, self).__init__()
    model_resnet = resnet50(pretrained=True) if resnet_name == 'resnet50' else resnet101(pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
    self.__in_features = model_resnet.fc.in_features

  def forward(self, x):
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features
  
  def parameters_list(self, lr):
    parameter_list = [
        {'params': self.parameters(), 'lr': lr * 0.1},
    ]
    return parameter_list

class Encoder(nn.Module):
    def __init__(self, input_size, mid_size, hidden_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, mid_size, bias = True)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(mid_size, hidden_size, bias = True)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(mid_size, hidden_size, bias = False)

        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        self.fc3.apply(init_weights)


    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x2 = self.bn2(self.fc2(x))
        x3 = self.fc3(x)
        x3 = F.softplus(x3)
        return x2, x3
        

class Decoder(nn.Module):
    def __init__(self, hidden_size, mid_size, output_size):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(hidden_size, mid_size, bias = True)
        self.fc2 = nn.Linear(mid_size, output_size, bias = True)
        self.relu1 = nn.ReLU()
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LinearClassifier, self).__init__()
        self.fc = weightNorm(nn.Linear(input_dim, nclass), name = 'weight')
        self.fc.apply(init_weights)
    def forward(self, x): 
        o = self.fc(x)
        return o    

class MyGuidedModel(nn.Module):
    def __init__(self, config):
        super(MyGuidedModel, self).__init__()
        self.encoder = Encoder(2048, config.diffusion.aux_cls.h_dim, config.diffusion.aux_cls.z_dim)
        self.s_decoder = Decoder(config.diffusion.aux_cls.z_dim, config.diffusion.aux_cls.h_dim, 2048)
        self.t_decoder = Decoder(config.diffusion.aux_cls.z_dim, config.diffusion.aux_cls.h_dim, 2048)
        self.classifier = LinearClassifier(config.diffusion.aux_cls.z_dim, config.data.num_classes)
        
    def forward(self, x, return_hidden = False):
        mu, sigma = self.encoder(x)
        z = reparameterize(mu, sigma)        
        z = z.rsample()
        out = self.classifier(z)
        if return_hidden:
            return out, z
        else:
            return out

    def parameters_list(self, lr):
        parameter_list = [
            {'params': self.encoder.parameters(), 'lr': lr * 1.0},
            {'params': self.s_decoder.parameters(), 'lr': lr * 1.0},
            {'params': self.t_decoder.parameters(), 'lr': lr * 1.0},
            {'params': self.classifier.parameters(), 'lr': lr * 1.0}
        ]
        return parameter_list

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x, coeff):
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1
    
    def parameters_list(self, lr):
        parameter_list = [
            {'params': self.parameters(), 'lr': lr * 1.0},
        ]

        return parameter_list