import os
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


import ssd_mobilenet_v1
def run_model(simulation_file='../../../simulation_files/sigma_128mses_64_bw.cfg', tiles_path='tiles/accumulation_buffer/128_mses/', sparsity_ratio=0.0, stats_path='', trained_weights='weights_tensorflow.pb', image_input='../tinycoco/000000037777.jpg'):
    net = ssd_mobilenet_v1.get_tf_pretrained_mobilenet_ssd(trained_weights, simulation_file, tiles_path, sparsity_ratio, stats_path)    # initialize SSD
    image = cv2.imread(image_input, cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded
    from matplotlib import pyplot as plt
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # View the sampled input image before transform
    #plt.figure(figsize=(10,10))
    #plt.imshow(rgb_image)
    #plt.show()

    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    #plt.imshow(x)
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)

    print('Execution finished successfully')

#run_model()
