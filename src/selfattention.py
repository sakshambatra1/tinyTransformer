import jax 
import jax.numpy as jnp 
from jax import random

def lin_projections(x: jnp.ndarray, key: jax.Array, d_model: int, head_dim: int):

    kq, kk, kv = random.split(key, 3)

    W_q = random.normal(kq, (d_model, head_dim))
    W_k = random.normal(kk, (d_model, head_dim))
    W_v = random.normal(kv, (d_model, head_dim))

    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    return Q, K, V

def softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray: 
    x_shifted = x - jnp.max(x, axis = axis , keepdims=True)
    exp_x = jnp.exp(x_shifted)
    sum_exp = jnp.sum(exp_x, axis = axis, keepdims=True)
    return exp_x/sum_exp

def scaled_dot_product_att(Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray: 
    d_k = Q.shape[-1]
    scores = (Q @ K.T)/jnp.sqrt(d_k)
    weights = softmax(scores, axis=1)
    out = weights @ V
    return out 

def init_mheadattn(key:jax.Array, d_model:int, head_dim:int, num_heads:int) -> jnp.ndarray:
    kq, kk, kv = random.split(key, 3)

    W_q = random.normal(kq, (d_model, num_heads*head_dim))
    W_k = random.normal(kk, (d_model, num_heads*head_dim))
    W_v = random.normal(kv, (d_model, num_heads*head_dim))
    return W_q, W_k, W_v

num_heads = 4
head_dim = 4
@jax.jit
def multihead_attn(x:jnp.ndarray, W_q:jax.Array, W_k:jax.Array, W_v:jax.Array) -> jnp.ndarray:

    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    batch_size, seq_len, _ = x.shape
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim)
    Q = Q.transpose(0,2,1,3)

    K = K.reshape(batch_size, seq_len, num_heads, head_dim)
    K = K.transpose(0,2,1,3)

    V = V.reshape(batch_size, seq_len, num_heads, head_dim)
    V = V.transpose(0,2,1,3)

    scores = jnp.matmul(Q, K.swapaxes(-2,-1))/jnp.sqrt(head_dim)

    weights = softmax(scores, axis=-1)

    out = weights @ V
    out = out.transpose(0,2,1,3).reshape(batch_size, seq_len, d_model)
    return out 

key = random.PRNGKey(0)
batch, seq, d_model, num_heads = 1, 4, 8, 4
head_dim = d_model // num_heads

x = random.normal(key, (batch, seq, d_model))
kq, kk, kv = random.split(key, 3)
W_q = random.normal(kq, (d_model, num_heads * head_dim))
W_k = random.normal(kk, (d_model, num_heads * head_dim))
W_v = random.normal(kv, (d_model, num_heads * head_dim))


print(jax.make_jaxpr(multihead_attn)(x, W_q, W_k, W_v))
print(multihead_attn.lower(x, W_q, W_k, W_v).compiler_ir('stablehlo').operation.get_asm())






    
    

    
