# attention mechanism => 
# input sentence = My name is Sreerag
#              My    name  is  Sreerag
# My        [  4      3     1     2   ]
# name      [  3      3     2     4   ]
# is        [  1      2     4     2   ]
# Sreerag   [  2      4     1     4   ]

# "Sreerag" and "name" are closely related. Hence vector 
# corresponding to "Sreerag" will incorporate some more context
# with respect to "name"


# Tranformer Overview (for sequence to sequence tasks)=>
# Input --simulatenously--> 
# Encoder ==> generate vectors for each word {ie, broken down word pieces or subwords} --simulatenously-->
# Decoder architecture ==> <start> token --> generate new token --> feed <start> + <first token> --back to decoder--> 
# {ie. Start by generating "Ente" ie. My then take "Ente" as input and generate "Peru" ie. Name and so on}

# Attention in Transformer ( Advantage: Parallel processing, Context aware)
# input -> encoding ->   ( )  ->  Vectors for individual words (contest unaware)  -> ENCODER (Multi-Head Attention -> Feed Forward) -> Another set of vector (context aware)
                #         ^
                #         |
                # Positional encoding
 
# ATTENTION MECHANISM
# Every word will have three vectors
    # a) Query vector (Q) : What I am looking for { [sequence length * d(k)] }
    # b) Key vector (K) : What I can offer { [sequence length * d(k)] }
    # c) Value vector (V) : What I actually offer { [sequence length * d(v)] }

import numpy as np
import math

L, d_k, d_v = 4, 8, 8  # L= Seq Length of input
q=np.random.randn(L, d_k) # Randomly generate using normal distribution
k=np.random.randn(L, d_k)
v=np.random.randn(L, d_v)



print ("Query vector (Q) :\n", q)
print ("Key vector (K) :\n", k)
print ("Value vector (V)\n", v)

# Ouput is an 8 * 1 vector for Q, K and V for each word

# self attention = softmax((Q * K^T / sqrt(d_k)) + M ) * V  
# Q * K^T => Every word look at every other word to see if they have higher affinity to a word

print(" Q * K^T => \n", np.matmul(q, k.T)) 

# Why we need sqrt(d_k) in denominator
# It is to stabilize value by minimizing the variance of Q * K^T

print("Variance: \n", q.var(), k.var(), np.matmul(q, k.T).var())

# while Q and K variance is in same range, the variance of q * K.T is out of range

scaled = np.matmul(q, k.T) / math.sqrt(d_k)
print("New Variance: \n", q.var(), k.var(), scaled.var())
print("Vector: \n", scaled)

# Masking: Important in decoder part so that we don't look at a future word when 
# trying to generate the context of current word 

mask = np.tril(np.ones((L, L)))
print("Mask:\n", mask)

# This ensures that the word can only look at previous words and not the next one
print("Mask + Scaled:\n", scaled + mask)
mask[mask == 0] = -np.infty
mask[mask == 1] = 0
print("Transformed Mask:\n", mask)
print("Scaled + Mask:\n",scaled + mask)
# This transformation lets us not to take any context from it
# Adding negative infinity is better for the next softmax operation

# Softmax =>
# converts a vector to probability distribution
# softmax = e^x(i)/ Σ e^x(j)

def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=-1)).T

attention = softmax(scaled + mask)

print("Attention:\n", attention)

# The application of mask lets allows to channel focus on the previous and current word and 
# gives accurate attention in the form of probability for each word

new_v = np.matmul(attention, v)

print("Previous Vector:\n", v)
print("New Vector:\n", new_v)


# Hence a single head would look like

import numpy as np
import math
def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=-1)).T

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scaled = np.matmul(q, k.T) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled + mask
    attention = softmax(scaled)
    out = np.matmul(attention, v)
    return out, attention

out, attention = scaled_dot_product_attention(q, k, v, mask=mask)

print("Attention:\n", attention)
print("Output:\n", out)


