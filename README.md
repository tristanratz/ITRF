
This is the Repo for the work WORK IN PROGRESS

# Using the model

To load the model simply use the model/ folder where the LLM and the reranker can be loaded.
Models will be automatically downloaded from huggingface.

# Running the inference

To just run the inference and interact with the pipeline, please use the main.py script.

# Retriever corpus

To recreate the corpos for this work 

1. follow the instructions in the [Meta ATLAS Repo](https://github.com/facebookresearch/atlas) to retrieve their wikipedia dump and load them under ./data/corpora/
2. Start the vector store docker container using the script at create_corpus/vector_store/
4. Please make sure to create a collection called "retriever" to disable indexing before uploading the batches and enable it afterwards. This can be done using the jupyter notebook to be found at create_corpus/.
3. Use the scripts provided in "create_corpus" to load the wikipedia as well as the CC dump into the vector database

The shell script will start four workers which each load transform split the dataset into chunks and embed as well as upload them to the vector store in batches of 1000 chunks.

For the loading of the complete ATLAS wikipedia dump consisting of about 33mio samples it took ~9,5h using 4x Nvidia RTX 3090.

# Recreating the dataset

For recreating the dataset you can run the script in create_dataset/.
Please note that for the notebook to work, the retriver corpus container must be already working.

The dataset can also be retrieved from [HuggingFace](https://huggingface.co/datasets/tristanratz/DATASET).

```python
from datasets import load_dataset

dataset = load_dataset("tristanratz/rarit", "llm")

```

Before running the dataset creation for the reranker make sure that you trained the LLM beforehand.

# Training the LLM model

For training the models please use the python scripts in train_llm/.
Please make sure to have the dataset at hand. 

To accelerate the training process and to use multiple GPUs consider configuring accelerate using
```bash
accelerate config --config_file <CONFIG_FILE>
```
We optimized and ran our training using deepspeed zero and multiple optimization techniques (more details see train_llm/README) to especially run on 4x RTX 3090.

To run the training for the 13b-qlora variant of our model use "train_llm/run_training.sh".

# Train the Reranker

Content to follow

# Evaluation

Use the script to be found in evaluation to evluate the different parts in 
