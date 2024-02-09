from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import models, QdrantClient

col_name = "retriever"

embedding = HuggingFaceBgeEmbeddings(model_name="../models/retriever/bge-base-en-v1.5", model_kwargs={"device": "cpu"})

# Create the retriever
client = QdrantClient(url="http://localhost:6333")

# Create the collection
vector_size = len(embedding.embed_query("Test query"))
client.create_collection(collection_name=col_name, 
                         vectors_config=models.VectorParams(
                            size=vector_size,  # Vector size is defined by used model
                            distance=models.Distance.COSINE,
                            on_disk=True,
                         ),
                         hnsw_config=models.HnswConfigDiff(on_disk=True),
                         # optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
                         optimizers_config=models.OptimizersConfigDiff(
                             indexing_threshold=0,
                         ),
    )