import torch

batch_size=1

conv1 = torch.nn.Conv1d(in_channels=64, out_channels=128,kernel_size=2) # Creates the operation
my_input = torch.randn(batch_size,64,32) # Creates a tensor of size batch_size*in_channels*x
out = conv1(my_input) #Applies the operation
print(out) # Prints the result
print(out.shape)
