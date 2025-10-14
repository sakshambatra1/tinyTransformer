import jax.numpy as jnp 
import jax 
from jax import random 
from selfattention import multihead_attn

def relu(x:jax.Array):
    return jnp.maximum(0,x)

class LayerNorm: 
    def __init__(self, dim, eps=1e-15): 
        self.eps = eps 
        self.gamma = jnp.ones((dim,))
        self.beta = jnp.zeros((dim,))

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(mean, axis = -1, keepdims=True)
        x_norm = (x-mean)/jnp.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta
        return out 

class FeedForwardNN: 
    def __init__(self, d_model:int, d_ff:int, key:jax.Array):
        k1, k2, k3, k4 = random.split(key, 4)
        self.W1 = random.normal(k1, (d_model, d_ff)) * jnp.sqrt(2/d_model)
        self.W2 = random.normal(k2, (d_ff, d_model)) * jnp.sqrt(2/d_ff)
        self.B1 = jnp.zeros((d_ff,))
        self.B2 = jnp.zeros((d_model,))

    def __call__(self, x):
        layer1 = (x @ self.W1) + self.B1
        layer2 = relu(layer1)
        layer3 = (layer2 @ self.W2) + self.B2
        return layer3
    
class encoderBlock: 
    def __init__(self, d_model:int, d_ff:int, num_heads:int, key:jax.Array):
        k1, k2, k3, k_ff = random.split(key, 4)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        
        self.W_q = random.normal(k1, (d_model, d_model)) *jnp.sqrt(2.0/ d_model)
        self.W_k = random.normal(k2, (d_model, d_model)) * jnp.sqrt(2.0/d_model)
        self.W_v = random.normal(k3,(d_model, d_model)) * jnp.sqrt(2.0/d_model)
        
        self.ffnn = FeedForwardNN(d_model, d_ff, k_ff)
        
        self.d_model = d_model
        self.num_heads = num_heads

    def __call__(): 


