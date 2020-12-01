import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.SimulatedLinear(4,8, '../simulation_files/maeri_128mses_64_bw.cfg', 'dogsandcats_tile_fc.txt', sparsity_ratio=0.0)
        self.real_fc1 = nn.Linear(4,8)
        self.real_fc1.weight=self.fc1.weight
        self.real_fc1.bias  = self.fc1.bias
    def forward(self, x):
        print('Printing input tensor from python')
        #print(x)
        #print(self.real_fc1.weight)
        x_sim = self.fc1(x)
        x_real = self.real_fc1(x)
        return x_sim, x_real
    
net = Net()  
print(net)
input_test = torch.randn((8,4)).view(8,4)
output_sim, output_real  = net(input_test)
print(output_sim)
print(output_real)


