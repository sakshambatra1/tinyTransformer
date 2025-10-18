import jax 
import jax.numpy as jnp 
from embeddings import make_embeddings, embed_tokens
from transformers import AutoTokenizer
from data import make_next_token_pair

tok = AutoTokenizer.from_pretrained("gpt2")

key = jax.random.PRNGKey(0)

vocab_size = tok.vocab_size
d_model = 512
embedding_matrix = make_embeddings(key, vocab_size, d_model)

input_ids, target_ids = make_next_token_pair(tok, "luka is the best player", max_len=32)

input_ids = jnp.array(input_ids)
target_ids = jnp.array(target_ids)

embs = embed_tokens(embedding_matrix, input_ids)

print("embeddings shape: ", embs.shape)

