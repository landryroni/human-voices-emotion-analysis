import torch
from torch.autograd import Variable
import math
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle
import torch
from fastprogress.core import format_time
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# torch.max(labels, 1)[1]
#criterion(outputs,target.view(1, -1))
# with loss = criterion(outputs,target.view(-1, 1))

def accuracy(outputs, labels):
    # labels = labels.view(-1, 1)
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, torch.max(labels, 1)[1])  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, torch.max(labels, 1)[1])  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(1, 3)),
            nn.Conv2d(64, 64, kernel_size=(1, 3)),
            nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.MaxPool2d((1, 2)))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 1)),
            nn.Conv2d(64, 128, kernel_size=(3, 1)),
            nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.MaxPool2d((2, 1)))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1,3)),
            nn.Conv2d(128, 128, kernel_size=(1, 3)),
            nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.MaxPool2d((1, 2)))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 1)),
            nn.Conv2d(128, 128, kernel_size=(3, 1)),
            nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.MaxPool2d((2, 1)))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 1)),
            nn.Conv2d(256, 256, kernel_size=(3, 1)),
            nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.MaxPool2d((2, 1)))
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 1)),
            nn.Conv2d(512, 512, kernel_size=(3, 1)),
            nn.BatchNorm2d(512),
            # nn.ReLU(),
            nn.MaxPool2d((2, 2)))


        # self.flatten = nn.Flatten()
        # # self.lstm = nn.LSTM(1024, 100, bidirectional=True)
        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x1 = self.layer1(x)
        print("x1:",x1.size())
        # x = x.permute(0,2,3,1)

        x2 = self.layer2(x1)
        print("x2:", x2.size())

        x3 = self.layer3(x2)
        print("x3:", x3.size())
        # x11 = x1 + x3
        # print("x11:", x11.size())

        x4 = self.layer4(x3)
        print("x4:", x4.size())
        # x22 = x2 + x4
        # print("x22:", x22.size())

        x5 = self.layer5(x4)
        print("x5:", x5.size())
        x6 = self.layer6(x5)
        print("x6:", x6.size())

        # x33 = x5 + x6
        # print("x33:", x33.size())
        x44 = x6 #+ x22 + x33
        print("x44:", x44.size())
        out61 = F.max_pool2d(x44, (1, 2))
        print(out61.size())
        out62 = F.avg_pool2d(x44, (1, 2))
        print(out62.size())
        x = torch.cat([out61, out62], dim=1)
        print(x.size())
        # x = x.permute(0, 3, 1, 2)
        # x = x.view(x.size(0), x.size(1), -1)
        # x = x.permute(1, 0, 2)
        # x, _ = self.lstm(x)

        # x = self.flatten(x)
        # print(x.size())
        # x = self.fc1(x)
        # print(x.size())
        # x = self.fc2(x)
        # print(x.size())
        return x #F.log_softmax(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        print("x_before_chanel",x.size())
        x_out = self.ChannelGate(x)
        print("x_out",x_out.size())
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
            print("x_out_fater_spacialgate",x_out.size())
        return x_out


def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.weight.data.fill_(1.)

class ShakeShake(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2, training=True):
        if training:
            alpha = torch.cuda.FloatTensor(x1.size(0)).uniform_()
            alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x1)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_()
        beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
        beta = Variable(beta)

        return beta * grad_output, (1 - beta) * grad_output, None

class FeatureLevelSingleAttention(ImageClassificationBase):

    def __init__(self):
        super(FeatureLevelSingleAttention, self).__init__()

        self.cnn = CNNModel()
        # self.channel = ChannelGate(64,reduction_ratio=16,pool_types=["avg","max"])
        self.channel_spatial = CBAM(300, reduction_ratio=4, pool_types=['avg', 'max'], no_spatial=False)
        self.fc1 = nn.Linear(1024*15,512)
        self.fc2 = nn.Linear(512,6)
        self.init_weights()
        # self.attention = Attention(
        #     hidden_units,
        #     hidden_units,
        #     att_activation='sigmoid',
        #     cla_activation='linear')
        #
        # self.fc_final = nn.Linear(hidden_units, classes_num)
        # self.bn_attention = nn.BatchNorm1d(hidden_units)
        #
        # self.drop_rate = drop_rate
        #
        # self.init_weights()

    def init_weights(self):
        init_layer(self.fc1)
        # init_bn(self.bn_attention)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """
        x2 = x.permute(0, 2, 3, 1)
        print("x.shape", x.size())
        print("x2.shape", x2.size())
        spatial_out = self.channel_spatial(x2)
        print("spatial_out", spatial_out.size())
        #out = ShakeShake.apply(x2, spatial_out, self.training)
        out  = x2 +  spatial_out
        print("out-shake", out.size())
        out = out.permute(0,3,1,2)
        print("out", out.size())
        cnn_out = self.cnn(out)
        print("cnn_out",cnn_out.size())
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        fc1 = self.fc1(cnn_out)
        print("fc1",fc1.size())
        # fc2 = self.fc2(fc1)
        output = F.softmax(self.fc2(fc1))#self.fc2(fc1) #F.softmax(self.fc2(fc1))
        print("output",output.size())
        # out = torch.cat([cnn_out,spatial_out])
        # drop_rate = self.drop_rate
        #
        # # (samples_num, hidden_units, time_steps, 1)
        # b1 = self.emb(input)
        #
        # # (samples_num, hidden_units)
        # b2 = self.attention(b1)
        # b2 = F.dropout(
        #     F.relu(
        #         self.bn_attention(b2)),
        #     p=drop_rate,
        #     training=self.training)
        #
        # # (samples_num, classes_num)
        # output = F.sigmoid(self.fc_final(b2))

        return output




@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# history = [evaluate(model, valid_dl)]
# print(history)
#
# epochs = 8
# max_lr = 0.01
# grad_clip = 0.1
# weight_decay = 1e-4
# opt_func = torch.optim.Adam

# %%time
# history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl,
#                              grad_clip=grad_clip,
#                              weight_decay=weight_decay,
#                              opt_func=opt_func)



# train_time='4:07'


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');


# plot_accuracies(history)

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');

# plot_losses(history)


def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');


# plot_lrs(history)
import jovian


def load_data():
    f = open('iemocap_features_40.pkl', 'rb')
    train_feature, train_labels, test_feature, test_seg_labels, valid_feature, valid_seg_labels, test_seg_nums, valid_seg_nums = pickle.load(f)
    return train_feature, train_labels, test_feature,test_seg_labels, valid_feature,valid_seg_labels, test_seg_nums,valid_seg_nums

# train_feature, train_labels, test_feature,test_seg_labels,
# valid_feature,valid_seg_labels, test_seg_nums,valid_seg_nums=pickle(f)

train_feature, train_labels, test_feature,test_seg_labels, valid_feature,valid_seg_labels, test_seg_nums,valid_seg_nums = load_data()

# print(train_feature[:,0,:,:])
x_train, y_train = torch.rand((train_feature.shape[0], train_feature.shape[1], train_feature.shape[2])), \
                   torch.rand((train_labels.shape[0], train_labels.shape[1]))  # data


class training_set(Dataset):
    def __init__(self,X,Y):
        self.X = X                           # set data
        self.Y = Y                           # set lables

    def __len__(self):
        return len(self.X)                   # return length

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]    # return list of batch data [data, labels]

batch_size = 100
training_dataset = training_set(train_feature, train_labels)
train_dl = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = training_set(valid_feature, valid_seg_labels)
valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


# second way same as class train_dataset function just use TensorDataset
test_feature = torch.from_numpy(test_feature)
test_seg_labels = torch.from_numpy(test_seg_labels)
training_dataset = TensorDataset(test_feature, test_seg_labels) #convert into  tensor by using torch.numpy
test_dl = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

# def show_batch(dl):
#     for images, labels in dl:
#         fig, ax = plt.subplots(figsize=(12, 12))
#         ax.set_xticks([]); ax.set_yticks([])
#         ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))
#         break

# for index, (data, label) in enumerate(valid_dl):
#         print(label.numpy())
#         print(data.shape)
#         plt.imshow(data.numpy()[0, :, :])
#         plt.show()
#
#         if index == 0:
#             break

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# device = get_default_device()
# device


if __name__ == "__main__":
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)
    # simple_resnet = to_device(SimpleResidualBlock(), device)

    model = to_device(FeatureLevelSingleAttention(), device)
    print(model)
    history = [evaluate(model, valid_dl)]
    print(history)

    epochs = 4
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam
    history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl,
                             grad_clip=grad_clip,
                             weight_decay=weight_decay,
                             opt_func=opt_func)
    train_time = '4:07'
    plot_accuracies(history)
    plot_losses(history)
    plot_lrs(history)

    torch.save(model.state_dict(), 'mine.pth')