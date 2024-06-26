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

# Add all the processed chunks from all the processes to the use the continiue point
parser.add_argument('--cdoc', type=int, default=0, help='If the process was interrupted, continue from this document')
parser.add_argument('--cchunk', type=int, default=0, help='If the process was interrupted, use this for counting the chunks')
args = parser.parse_args()

process = args.process

cdoc = args.cdoc
cchunk = args.cchunk

if args.tofile:
    # Put the output in a file
    f = open(f"../data/process_output/cc_dump-{process}.txt", "w")
    sys.stdout = f

# Load wikipedia corpus (in steps of 1000 documents)
documents_count = 0 # Total nr of documents loaded
total_count = cdoc*4 # Total nr of documents processed (over all processes)
chunks_count = cchunk # Total nr of chunks loaded
start = time.time()

def len_func(example):
    return len(example.split())

# Create retriever
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=0,
    length_function=len_func,
    is_separator_regex=False,
)
device = "cuda:" + str(process)
embedding = HuggingFaceBgeEmbeddings(model_name="../models/retriever/bge-base-en-v1.5", model_kwargs={ "device": device })

# Create the retriever
client = QdrantClient(url="http://localhost:6333")

db = Qdrant(client, 
            collection_name="retriever",
            embeddings=embedding,
            )



def index_doc(docs, nr_documents):
    global documents_count
    global total_count
    global chunks_count
    global start
    global db
    global text_splitter
    global embedding
    global client

    process_start = time.time()

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
    print(f"Loaded documents {nr_documents} (total: {documents_count} [frac: {round(documents_count/total_count, 2)}%]), with {len(docs)} chunks (total: {chunks_count}), {clock} elapsed (embedding: {time.time()-process_start}s)")

def main(process=0):
    global documents_count
    global total_count
    global chunks_count
    global start
    global db
    global cdoc
    global cchunk
    global text_splitter
    global embedding
    global client

    dataset = load_dataset('oscar', "unshuffled_deduplicated_en", split='train', streaming=True)
    shuffled_dataset = iter(dataset.shuffle(buffer_size=100_000, seed=2024))

    documents = [] # Buffer of split documents
    chunk_nr = 0 # Nr of chunks which are in the buffer
    doc_nr = 0 # Nr of documents which are in the buffer

    # Continue from a certain point
    if cdoc > 0:
        print(f"Continuing from document {cdoc}...")
        while documents_count < cdoc:
            next(shuffled_dataset)
           
            # Add for the total count
            if total_count % 4 == process:
                documents_count += 1
            
                if documents_count % 300_000 == 0:
                    elapsed = (time.time() - start)
                    clock = str(timedelta(seconds=elapsed))
                    print(f"Fast forwarding... {documents_count} documents reached (elapsed: {clock})")
            
            total_count += 1

        print(f"---- Contiue document ({cdoc}) reached ----")

    print("Loading documents...")
    while chunks_count < (150_000_000 / 4):            
        sample = next(shuffled_dataset)
        
        # Split document into chunks  
        # Use only 1/4 of the documents for each process
        if total_count % 4 == process:
            docs = text_splitter.create_documents([sample["text"]], [{"id": sample["id"], "src": "oscar"}])
            documents.extend(docs)
            chunk_nr += len(docs)
            doc_nr += 1
            documents_count += 1

        total_count += 1

        if chunk_nr > 1_000:
            chunks_count += chunk_nr
            index_doc(documents, doc_nr)

            # results = await db.aadd_documents(new_docs)
            # print(results)
            documents = []
            chunk_nr = 0
            doc_nr = 0
                
    db.add_documents(documents) 

    print("---------------------------------------------")
    print()
    elapsed = (time.time() - start)
    clock = str(timedelta(seconds=elapsed))
    print(f"Loaded {documents_count} documents, {chunks_count} chunks, {clock} elapsed")
    print()
    print("---------------------------------------------")    

if __name__ == "__main__":
    main(process=process)
    