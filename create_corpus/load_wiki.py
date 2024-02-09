from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from json import loads
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import models, QdrantClient
import time
from datetime import timedelta
import sys
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--process', type=int, default=0, help='The number of the process')
parser.add_argument('--tofile', type=bool, default=False, help='Output the results to the console')
args = parser.parse_args()

process = args.process

if args.tofile:
    # Put the output in a file
    f = open(f"../data/process_output/wiki_dump-{process}.txt", "w")
    sys.stdout = f

# Load wikipedia corpus (in steps of 1000 documents)
documents_count = 0
chunks_count = 0
start = time.time()

def len_func(example):
    return len(example.split())

# Create retriever
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=200,
    chunk_overlap=0,
    length_function=len_func,
    is_separator_regex=False,
)
device = "cuda:" + str(process)
# tokenizer = LlamaTokenizer.from_pretrained("../models/llama7b", device_map=device) # Tokenizer not used for text splitting
embedding = HuggingFaceBgeEmbeddings(model_name="../models/retriever/bge-base-en-v1.5", model_kwargs={ "device": device })

# Create the retriever
client = QdrantClient(url="http://localhost:6333")

# Optional: Delete the collection if it already exists
# client.delete_collection(collection_name="retriever")
# client.create_collection(collection_name="retriever", vectors_config=models.VectorParams(
#         size=768,  # Vector size is defined by used model
#         distance=models.Distance.COSINE,
#     ))

db = Qdrant(client, 
            collection_name="retriever",
            embeddings=embedding,
            )



def index_doc(documents, metadatas):
    global documents_count
    global chunks_count
    global start
    global db
    global text_splitter
    global tokenizer
    global embedding
    global client

    process_start = time.time()
    docs = text_splitter.create_documents(documents, metadatas)
    
    # Add documents to the database
    success = False
    while not success:
        try:
            success = db.add_documents(docs)
        except Exception as e:
            print("Error while adding documents to the database")
            print(e)
            print("Waiting for 120s for database to be ready...")
            time.sleep(120)

    elapsed = (time.time() - start)
    clock = str(timedelta(seconds=elapsed))
    chunks_count += len(docs)
    print(f"Loaded documents {len(documents)} (total: {documents_count}), with {len(docs)} chunks (total: {chunks_count}), {clock} elapsed (process: {time.time()-process_start}s)")

def main(process=0):
    global documents_count
    global chunks_count
    global start
    global db
    global text_splitter
    global tokenizer
    global embedding
    global client

    with open("../data/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl", mode="r") as f:
        documents = []
        metadatas = []
        tasks = []
        print("Loading documents...")
        for line in f:
            
            json_line = loads(line)

            # Split document into chunks  
            if documents_count % 4 == process:
                documents.append(json_line["text"])
                metadatas.append({"title": json_line["title"], "src": "wiki", "section": json_line["section"]})  
            documents_count += 1

            if documents_count % 4_000 == 0:
                index_doc(documents, metadatas)

                # results = await db.aadd_documents(new_docs)
                # print(results)
                documents = []
                metadatas = []
                

        new_docs = text_splitter.create_documents(documents, metadatas)            
        chunks_count += len(new_docs)
        db.add_documents(new_docs)
        

    print("---------------------------------------------")
    print()
    elapsed = (time.time() - start)
    clock = str(timedelta(seconds=elapsed))
    print(f"Loaded {documents_count} documents, {chunks_count} chunks, {clock} elapsed")
    print()
    print("---------------------------------------------")    

if __name__ == "__main__":
    main(process=process)