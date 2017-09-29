"""
autograd package provides automatic differentiation for all operations on Tensors.
"""

import torch
from torch.autograd import Variable

"""
autograd.Variable wraps around Tensor and 
supports (almost) all ops defined on it.

One can directly call .backward() and have
all gradients calculated automatically.
"""
x = Variable(torch.ones(2,2), requires_grad=True)
print(x)

y = x + 2
print(y)

"""
There is also another important class called Function
Variable and Function are interconnected and build up
an acyclic graph, that encodes complete history of
computation. Each variable has .grad_fn attribute that
references a Function that has created the Variable
(For user created Variable, grad_fn is None)
"""

z = y * y * 3
out = z.mean()
print(z, out)

"""
GRADIENTS
"""

out.backward()

print(x.grad)

x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)

print(x.grad)
