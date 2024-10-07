import torch
import dop_cuda_demo

import time

n = 2
m = 4

######
mat_1 = torch.rand(n,m,device='cuda:0')
mat_2 = torch.rand(n,m,device='cuda:0')
mul = 2.0

######

t = time.time()
out = dop_cuda_demo.dop(mat_1,mat_2, mul)
print('cuda time', time.time()-t, 's')


######
t = time.time()
out_py = mat_1*mul + mat_2
print('pytorch time', time.time()-t, 's')

print('results all close', torch.allclose(out, out_py))