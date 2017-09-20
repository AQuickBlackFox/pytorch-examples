"""
Code from
http://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
"""

from __future__ import print_function
import torch

# Constructs a 5x3 matrix, uninitialized
x = torch.Tensor(5, 3)
print(x)


# Constructs a randomly initialized matrix
x = torch.rand(5, 3)
print(x)
# Get its size
print(x.size())

"""
Operations
"""

# Addition, syntax 1
y = torch.rand(5, 3)
print(x + y)

# Addition, syntax 2
print(torch.add(x,y))

# Addition, giving an output tensor
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

# Addition in place
y.add_(x)
print(y)

# Can use numpy like indexing
print(x[:,1])

"""
Numpy bridge
"""
# converting torch tensor to numpy array
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

# The numpy version of torch tensor is just a reference
# Changing torch tensor changes numpy array
a.add_(1)
print(a)
print(b)

# Converting numpy array to torch tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# Cuda Tensors
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y
