# MultiHead attention
                                                                    # All other words as well
# "name" vector (512 x 1) --> q, k, v (512 x 1 each)                            | | |
                         #   |  |  |
                        #  8 * 64 x 1 sub vectors each ( each is one head)  -> Attention Unit  -> 8 *  Attention matrix --> Output vectors


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


sequence_length = 4
batch_size = 1
input_dim = 512
d_model = 512
x = torch.randn( (batch_size, sequence_length, input_dim))

print(x.size())

qkv_layer = nn.Linear(input_dim, 3 * d_model) # create concatenated q v and k with 8 attention heads

qkv = qkv_layer(x)

print(qkv.shape)  # one batch 4 words and each with 1536 size

# Visualize distribution 
# y_val = torch.histc(qkv, bins=200, min=-3, max=3)
# x_val = np.arange(-1, 1, 0.01) * 3
# plt.bar(x_val, y_val, align='center', color='darkcyan')
# plt.title('qkv distribution')

# plt.show()


num_heads = 8
head_dim = d_model  // num_heads
qkv = qkv.reshape(batch_size, sequence_length, num_heads, 3 * head_dim)
qkv = qkv.permute(0, 2, 1, 3)

print(qkv.shape) 

q, k, v = qkv.chunk(3, dim=-1)

print(q.shape)
print(k.shape)
print(v.shape)

d_k = q.size()[-1]
scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # cannot use k.T since these are 4 dimensional tensors rather than 2 dimensional matrices
# transpose sequence length and head dimension size
print(scaled.shape)

# Masking

mask = torch.full(scaled.size(), float('-inf'))
mask = torch.triu(mask, diagonal=1)

print(mask[0][1]) # mask for input to a single head

scale_mask = (scaled + mask)[0][0]
print(scale_mask) # scale for one head

scaled += mask

attention = F.softmax(scaled, dim=-1)

print(attention.shape)
 
values = torch.matmul(attention, v) # new value vectors that are more context aware

print(values.shape)

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

values, attention = scaled_dot_product(q, k, v, mask=mask)
print(attention.shape)
print(attention[0][0])

values = values.reshape(batch_size, sequence_length, num_heads * head_dim)  # combines all heads together
# print(values.size())

linear_layer = nn.Linear(d_model, d_model) # Feed forward layer 512 x512

out = linear_layer(values)

print(out.shape)


# Hence we get

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size()
        print(f"x.size(): {x.size()}")
        qkv = self.qkv_layer(x)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3)
        print(f"qkv.size(): {qkv.size()}")
        q, k, v = qkv.chunk(3, dim=-1)
        print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}, ")
        values, attention = scaled_dot_product(q, k, v, mask)
        print(f"values.size: {values.size()}, attention.size: {attention.size()}")
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim) 
        print(f"values.size: {values.size()}")
        out = self.linear_layer(values)
        print(f"out.shape: {out.shape}")
        return out

input_dim = 1024
d_model = 512
num_heads = 8
batch_size = 30
sequence_length = 5
x =torch.randn((batch_size, sequence_length, input_dim))

model = MultiheadAttention(input_dim, d_model, num_heads)
out = model.forward(x)