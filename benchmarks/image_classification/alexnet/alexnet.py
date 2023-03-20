import torch
import torch.nn as nn
from utils import load_state_dict_from_url


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        '''self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )'''

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        '''self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )'''

        self.features = nn.Sequential(
            nn.SimulatedConv2d(3, 64, kernel_size=11, path_to_arch_file='maeri_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_conv1.txt', sparsity_ratio=0.90, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.SimulatedConv2d(64, 192, kernel_size=5,path_to_arch_file='maeri_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_conv2.txt', sparsity_ratio=0.90, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.SimulatedConv2d(192, 384, kernel_size=3, path_to_arch_file='maeri_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_conv3.txt', sparsity_ratio=0.90, padding=1),
            nn.ReLU(inplace=True),
            nn.SimulatedConv2d(384, 256, kernel_size=3, path_to_arch_file='maeri_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_conv4.txt', sparsity_ratio=0.90,padding=1),
            nn.ReLU(inplace=True),
            nn.SimulatedConv2d(256, 256, kernel_size=3, path_to_arch_file='maeri_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_conv5.txt', sparsity_ratio=0.90, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.SimulatedLinear(256 * 6 * 6, 4096, path_to_arch_file='maeri_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_fc6.txt', sparsity_ratio=0.90),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.SimulatedLinear(4096, 4096,path_to_arch_file='maeri_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_fc7.txt', sparsity_ratio=0.90),
            nn.ReLU(inplace=True),
            nn.SimulatedLinear(4096, num_classes, path_to_arch_file='maeri_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_fc8.txt', sparsity_ratio=0.90),
        )



        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet_model(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                           progress=progress)
        model.load_state_dict(state_dict)
    return model


#alex_model = alexnet(pretrained=True)
