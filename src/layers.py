import jax.numpy as jnp 
import jax 
from jax import random 
from selfattention import multihead_attn

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