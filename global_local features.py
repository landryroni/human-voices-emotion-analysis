import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .resnet import resnet50


class Model(nn.Module):
  def __init__(self, local_conv_out_channels=128, num_classes=None):
    super(Model, self).__init__()
    self.base = resnet50(pretrained=True)
    planes = 2048
    self.local_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
    self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
    self.local_relu = nn.ReLU(inplace=True)

    if num_classes is not None:
      self.fc = nn.Linear(planes, num_classes)
      init.normal(self.fc.weight, std=0.001)
      init.constant(self.fc.bias, 0)

  def forward(self, x):
    """
    Returns:
      global_feat: shape [N, C]
      local_feat: shape [N, H, c]
    """
    # shape [N, C, H, W]
    feat = self.base(x)
    global_feat = F.avg_pool2d(feat, feat.size()[2:])
    # shape [N, C]
    global_feat = global_feat.view(global_feat.size(0), -1)
    # shape [N, C, H, 1]
    local_feat = torch.mean(feat, -1, keepdim=True)
    local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
    # shape [N, H, c]
    local_feat = local_feat.squeeze(-1).permute(0, 2, 1)

    if hasattr(self, 'fc'):
      logits = self.fc(global_feat)
      return global_feat, local_feat, logits

    return global_feat, local_feat


class initialize_model(nn.Module):
    def __init__(self, num_classes=10, feature_extract=True, use_pretrained=True):
        super(initialize_model, self).__init__()

        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        self.features_i = model_ft.features
        self.features_ii = model_ft.features

        self.classifier = model_ft.classifier
        self.classifier[1] = nn.Linear(2 * model_ft.classifier[1].in_features, model_ft.classifier[1].out_features)
        self.classifier[6] = nn.Linear(model_ft.classifier[6].in_features, num_classes)

    def forward(self, input_i, input_ii):
        output_i = self.features_i(input_i)
        output_i = output_i.view(output_i.size(0), -1)
        output_ii = self.features_ii(input_ii)
        output_ii = output_ii.view(output_ii.size(0), -1)

        output = torch.cat((output_i, output_ii), dim=1)
        output = self.classifier(output)
        return output