docker run -p 6333:6333 -p 6334:6334  \
    -v $(pwd)/../../data/vector_store/storage:/qdrant/storage \
    -v $(pwd)/../../data/vector_store/snapshots:/qdrant/snapshots \
    -v ./config.yaml:/qdrant/config/production.yaml \
    --name vector_store \
    --detach \
    qdrant/qdrant 

# --ulimit nofile=10000:10000 \
# -e QDRANT_ALLOW_RECOVERY_MODE=true \

python3 ./create_collection.py