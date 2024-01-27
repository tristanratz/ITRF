
This is the Repo for the work WORK IN PROGRESS

# Retriever corpus

To recreate the corpos for this work 

1. follow the instructions in the [Meta ATLAS Repo](https://github.com/facebookresearch/atlas) to retrieve their wikipedia dump
2. setup the qdrant/qdrant image with open ports 6333 and 6334. 
3. use the scripts provided in "create_corpus" to load the wikipedia as well as the CC dump into the vector database

For the loading of the complete ATLAS wikipedia dump consisting of about 33mio samples it took ~9h using 4x Nvidia RTX 3090.

# Recreating the dataset

