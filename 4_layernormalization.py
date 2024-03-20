# Add and Norm

# It takes the multihead matrix and positional encoded matrix 
# This is to ensure stronger information signal
# This prevents the vanishing gardients during backpropogation. Thus ensure the model keeps learning
# Thus we add and norm the two matrices

# Layer Normalization
# Activation of neurons is wide range of +ve and -ve values
# Normalization encapsulates this range to be close to zero
# This allow stable training and faster to get to optimal values
# Layer Normalization ensure that activation of neurons in every layer, will have a median of zero and standard deviation of 1
 
 
 # X' = f[W^T * x + b]
 # Y = 	γ [ (X' - mean) / s_d ] + β   => 	γ, β same of a layer
 
import torch
from torch import nn
 
inputs = torch.Tensor([[[0.2, 0.1, 0.3], [0.5, 0.1, 0.1]]])
B, S, E = inputs.size()
inputs = inputs.reshape(S, B, E)
print(inputs.size())

parameter_shape = inputs.size()[-2:]
gamma = nn.Parameter(torch.ones(parameter_shape))
beta = nn.Parameter(torch.zeros(parameter_shape))

print(gamma.size())
print(beta.size())

dims = [-(i + 1) for i in range(len(parameter_shape))]

print(dims)

mean = inputs.mean(dim=dims, keepdim=True)
print(mean.size())
print(mean)

var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
epsilon = 1e-5   # so that denominator does not become zero
std = (var + epsilon).sqrt()
print(std)

y = (inputs - mean) / std
print(y)

out = gamma * y + beta
print(out)


# Hence we get

import torch
from torch import nn
 
class LayerNormalization():
    def __init__(self, parameters_shape, eps=1e-5):
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))
        
    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        print(f"Mean: \n ( {mean.size()}): \n {mean}")
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.esp).sqrt()
        print(f"Standard Deviation: \n ( {std.size()}): \n {std}")
        y = (inputs - mean) / std
        print(f"Y: \n ( {y.size()}): \n {y}")
        out = self.gamma * y + self.beta
        return out