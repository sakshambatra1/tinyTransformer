import jax
import jax.numpy as jnp
from jax import random, make_jaxpr

# ------------------ softmax ------------------
def softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    x_shifted = x - jnp.max(x, axis=axis, keepdims=True)
    exp_x = jnp.exp(x_shifted)
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)

# ------------------ attention core ------------------
@jax.jit
def multihead_attn_jit(x, W_q, W_k, W_v, head_dim, num_heads):
    batch_size, seq_len, embed_dim = x.shape

    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    scores = jnp.matmul(Q, K.swapaxes(-2, -1)) / jnp.sqrt(head_dim)
    weights = softmax(scores, axis=-1)
    out = jnp.matmul(weights, V)
    return out

# ------------------ inspection helper ------------------
def inspect_attention():
    key = random.PRNGKey(0)
    batch_size, seq_len, embed_dim = 1, 10, 512
    num_heads, head_dim = 8, 64
    kq, kk, kv = random.split(key, 3)

    W_q = random.normal(kq, (embed_dim, num_heads * head_dim))
    W_k = random.normal(kk, (embed_dim, num_heads * head_dim))
    W_v = random.normal(kv, (embed_dim, num_heads * head_dim))
    x = random.normal(key, (batch_size, seq_len, embed_dim))

    # ----------------- JAXPR -----------------
    print("="*30)
    print("JAXPR (High-Level IR)")
    print("="*30)
    jaxpr = make_jaxpr(multihead_attn_jit)(x, W_q, W_k, W_v, head_dim, num_heads)
    print(jaxpr)

    # ----------------- StableHLO -----------------
    print("\n" + "="*30)
    print("StableHLO (MLIR Dialect)")
    print("="*30)
    lowered = multihead_attn_jit.lower(x, W_q, W_k, W_v, head_dim, num_heads)
    print(lowered.compiler_ir(dialect="stablehlo").as_text()[:1500] + "\n[...truncated...]")

    # ----------------- XLA HLO -----------------
    print("\n" + "="*30)
    print("XLA HLO (High-Level Optimizer IR)")
    print("="*30)
    print(lowered.compiler_ir(dialect="hlo").as_text()[:1500] + "\n[...truncated...]")

if __name__ == "__main__":
    inspect_attention()
