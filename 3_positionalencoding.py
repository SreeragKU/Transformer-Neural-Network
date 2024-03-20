# Positional Encoding

# For i = dimension index, d_model = Embedding Length and pos = psotion of word in sequence

# PE(pos, 2i) = sin(pos/10000^(2i/d_model))
# PE(pos, 2i+ 1) = cos(pos/10000^(2i/d_model))

# Why PE? => Periodicity (sin and cos repeats), Constrained values ( +1 to -1 constrained), Easy to extrapolate for long sequences

import torch
import torch.nn as nn

max_sequence_length = 10
d_model = 6

even_i = torch.arange(0, d_model, 2).float()
print(even_i)

even_denominator = torch.pow(10000, even_i/d_model)
print(even_denominator)

odd_i = torch.arange(1, d_model, 2).float()
print(odd_i)

odd_denominator = torch.pow(10000, (odd_i - 1)/d_model)
print(odd_denominator)

denominator = even_denominator  # since even and odd denominators are same

positon = torch.arange(max_sequence_length, dtype=torch.float).reshape(max_sequence_length, 1)
print(positon)

even_PE = torch.sin(positon / denominator)
odd_PE = torch.cos(positon / denominator)

print(even_PE)
print(odd_PE)

stacked = torch.stack([even_PE, odd_PE], dim=2)
print(stacked.shape)

PE = torch.flatten(stacked, start_dim=1, end_dim=2)
print(PE)

# Hence we get

import torch
import torch.nn as nn

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        
    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i / self.d_model)
        positon = torch.arange(self.max_sequence_length, dtype=torch.float).reshape(self.max_sequence_length, 1)
        even_PE = torch.sin(positon / denominator)
        odd_PE = torch.cos(positon / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE
    
pe = PositionEncoding(d_model=6, max_sequence_length=10)
pe.forward()


        
