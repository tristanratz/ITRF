# Base on HuggingFace Adapter
# https://api.python.langchain.com/en/stable/_modules/langchain/embeddings/huggingface.html#HuggingFaceEmbeddings

# Test if embed documents is called in case of bulk call
# This would pose a problem with our approach (DRAGON is asymetric)

from langchain.schema.embeddings import Embeddings
import torch
from transformers import AutoTokenizer, AutoModel

class DRAGON(Embeddings):

    def __init__(self, path="./models/retriever/dragon-plus"):
        
        self.tokenizer = AutoTokenizer.from_pretrained(path + "-query-encoder")
        self.query_encoder = AutoModel.from_pretrained(path + "-query-encoder")
        self.context_encoder = AutoModel.from_pretrained(path + "-context-encoder")

    def embed_query(self, query):
        # Apply tokenizer
        query_input = self.tokenizer(query, return_tensors='pt')
        
        # Compute embeddings: take the last-layer hidden state of the [CLS] token
        return self.query_encoder(**query_input).last_hidden_state[:, 0, :]
    
    def embed_documents(self, contexts):
        # Apply tokenizer
        ctx_input = self.tokenizer(contexts, padding=True, truncation=True, return_tensors='pt')

        # Compute embeddings: take the last-layer hidden state of the [CLS] token
        return self.context_encoder(**ctx_input).last_hidden_state[:, 0, :]

if __name__ == "__main__":
    embedding = DRAGON()

    # We use msmarco query and passages as an example
    query =  "Where was Marie Curie born?"
    contexts = [
        "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
        "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
    ]

    query_emb = embedding.embed_query(query)    
    ctx_emb = embedding.embed_documents(contexts)

    # Compute similarity scores using dot product
    score1 = query_emb @ ctx_emb[0]  # 396.5625
    score2 = query_emb @ ctx_emb[1]  # 393.8340

    print(score1)
    print(score2)
