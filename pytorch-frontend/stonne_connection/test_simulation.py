import torch
import torch_stonne

a=torch.randn(6,5)
b=torch.randn(5,5)
r=torch_stonne.simulated_matmul("", a,  b, "../../simulation_files/maeri_128mses_128_bw.cfg", "test_tile.txt", 0)
print(r)
print(torch.matmul(a,b))
