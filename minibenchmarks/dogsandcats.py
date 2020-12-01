import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.SimulatedConv2d(4,4,4,'../simulation_files/tpu_16_16_pes.cfg', 'dogsandcats_tpu_tile.txt', sparsity_ratio=0.0, groups=4) # Number of input feature maps (input channels), number of output feature maps (output channels), filter dimension size
        #print(self.conv1.weight)
        self.real_conv1 = nn.Conv2d(4,4,4, groups=4)
        self.real_conv1.weight=self.conv1.weight 
        self.real_conv1.bias = self.conv1.bias
        print(self.conv1.bias.shape)
        print("Weight size is ")
        print(self.real_conv1.weight.shape)
    def forward(self, x):
        sim_x = self.conv1(x)
        real_x = self.real_conv1(x)
        return sim_x, real_x
    
net = Net()  
print(net)
input_test = torch.randn(4,32,32).view(-1,4,32,32)
output_sim, output_real  = net(input_test)
#print('Output simulated shape is ', output_sim.shape)
#print('Real tensor shape is ', output_real.shape)
print('Test value')
#print(output_sim)
#print(output_real)
#print(torch.eq(output_sim, output_real))
print(torch.all(torch.lt(torch.abs(torch.add(output_sim, -output_real)), 1e-4)))
print(output_sim[0][0][1][0])
print(output_real[0][0][1][0])
