import jax 
import jax.random as random
import jax.numpy as jnp 

key = random.PRNGKey(0)

vocab_size = 50257
embed_dim = 512 
shape = (vocab_size, embed_dim)

embedding_matrix = random.normal(key, shape)

print(embedding_matrix)

