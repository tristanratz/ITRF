from datasets import Dataset, load_dataset
from qdrant_client import models, QdrantClient
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Qdrant
import time
from datetime import timedelta
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import numpy as np
import argparse
import hashlib

import sys
import os
sys.path.append(os.path.abspath('../'))

from model.llm import LLM

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=7500, help='The number of samples per dataset')
parser.add_argument('--collection', type=str, default="retriever", help='Collection name in the vector database')
parser.add_argument('--k', type=int, default=100, help='The number of contexts per sample')
parser.add_argument('--options', type=bool, default=True, help='If the multiple choice options should be part of query')
parser.add_argument('--continue_point', type=str, default=None, help='Conitnue from this dataset.')
parser.add_argument('--sample_skip', type=int, default=0, help='Conitnue from this point in the dataset.')

args = parser.parse_args()

#######
# Config variables
n = args.n
col_name = args.collection
k=args.k
options_enabled = args.options
continue_point = args.continue_point
seed = 4048
data_path = '../data/dataset/itrf_evaluation_retrieval_2'
#######

datasets_evaluation = ["cais/mmlu", "natural_questions", "mandarjoshi/trivia_qa", "hotpot_qa"]

task_list = []
task_list.extend(datasets_evaluation)

tmp = []

# Remove all the datasets before continue point
cp = False
for d in task_list:
    if d == continue_point:
        cp = True
    if cp:
        tmp.append(d)
if continue_point in task_list:
    itrf = pd.read_parquet(f"{data_path}.parquet")
    task_list = tmp
else:
    itrf = DataFrame(columns=["split", "query", "prediction", "context", "src", "id", "context_src", "context_id", "original_context", "task", "domain"])

print(task_list)

itrf_dataset_buffer = []
total = len(itrf)

sentinel = object() # Used to check if the iterators are empty

embedding = HuggingFaceBgeEmbeddings(model_name="../models/retriever/bge-base-en-v1.5", model_kwargs={"device": "cuda:1"})

# Create the retriever
client = QdrantClient(url="http://localhost:6333")
db = Qdrant(client, 
            collection_name=col_name,
            embeddings=embedding,
            )


# llm = LLM(size=7, quantized=False, adapter=False, model_path="llmware/dragon-llama-7b-v0")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return x / np.sum(x)

def search(query: str, k: int = 3):
    success = False
    while not success:
        try:
            results = db.similarity_search_with_score(query, k=k)
            if results:
                success = True 
        except:
            print(f"Error with example {query}, retrying in 0.2s")
            time.sleep(0.2)
    return results

# def rerank(query: str, contexts: list, k: int = 3):

# This function creates an example with the query and the prediction and the top k results
def make_example(query: str, ground_truth:str, dataset_name:str, example_id = None, k: int = 3, split = "validation", retrieval = True, task="", domain=""):
    contexts = []
    if retrieval:
        # Search for the query
        results = search(query, k=k)

        # Get the softmax of the scores
        retriever_softmax = softmax([result[1] for result in results])
        # context_texts = [llm.format_llmware_prompt(query, result[0].page_content) for result in results]
        # llm_scores = llm.to_tokens_and_logprobs(context_texts, [ground_truth] * k)[1]
        # llm_softmax = softmax(llm_scores)

        # Get the text of the results
        contexts = [{
            "text": result[0].page_content, 
            "src": result[0].metadata["src"] if "src" in result[0].metadata.keys() else "unknown", 
            "id": str(result[0].metadata["id"] if "id" in result[0].metadata.keys() else result[0].metadata["title"]),
            "retriever_score": result[1],
            # "llm_score": lscore,
            "retriever_softmax": rsoft, 
            # "llm_softmax": lsoft, 
            # "llm_weighted_softmax": rsoft * lsoft, 
            "original_context": False } 
            # for result, rsoft, lsoft, lscore in zip(results, retriever_softmax, llm_softmax, llm_scores)]
            for result, rsoft in zip(results, retriever_softmax)]

    return { 
        "split": split, 
        "query": query, 
        "ground_truth": ground_truth, 
        "contexts": contexts,
        "src": dataset_name, 
        "id": str(example_id),
        "task": task,
        "domain": domain, 
        }

def save_example(i, start, last_time, example, dname, force=False):
    save_examples(i, start, last_time, [example], dname, force=force)

def save_examples(i, start, last_time, examples, dname, force=False):
    global itrf_dataset_buffer
    global itrf
    global total
    # Save the dataset to a file
    itrf_dataset_buffer.extend(examples)
    total += len(examples)

    if i % 100 == 0 or force:
        current_time = time.time()
        print(f"Processed {i} {dname} examples, time: {str(timedelta(seconds=(last_time - start)))}, last 100 in {str(timedelta(seconds=(current_time - last_time)))}, total: {total} examples")
        last_time = current_time
        if len(itrf_dataset_buffer) > 0:
            if itrf.empty:
                itrf = DataFrame(itrf_dataset_buffer)
            else:
                df = DataFrame(itrf_dataset_buffer)
                itrf = pd.concat([itrf, df])
            itrf_dataset_buffer.clear()
        if force:
            ###
            # Clean the dataset
            ####

            # Remove duplicates
            itrf.drop_duplicates(inplace=True)

            # Repair index
            index = range(len(itrf))
            itrf.index = index

            # Remove examples with na fields
            itrf.dropna(inplace=True)

            # Remove the examples with unknown answer
            itrf = itrf[itrf["ground_truth"] != "I don't know."]
        itrf.to_parquet(f"{data_path}.parquet")
        # itrf.to_csv(f"{data_path}.csv", escapechar='\\')

def skip(dataset, dname):
    global continue_point
    
    if dname == continue_point:
        continue_point = None
        for _ in range(args.sample_skip+1):
            next(dataset)
        print("Continue from", args.sample_skip+1, "to", n)
        return range(args.sample_skip+1, n)
    return range(n)

##################
# Evaluation set
##################

#####
# MMLU
#####

dname = datasets_evaluation[0]
if dname in task_list:
    dataset = load_dataset(dname, "all", split="test")
    shuffled = iter(dataset.shuffle(seed=seed))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        options = ""
        if options_enabled:
            options =  "\nOptions: " + ", ".join([f"{idx+1}) {c}" for idx, c in enumerate(example["choices"])]) + "\n"
        query = example["question"] + options

        answer = ""
        if options_enabled:
            answer = str(example["answer"]+1) + ") " # + str(example["choices"][example["answer"]])

        answer += str(example["choices"][example["answer"]])

        prediction = answer
        example_id = hashlib.md5((query + "_" + prediction).encode()).hexdigest()
        example = make_example(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="mc", domain="trivia")
        
        save_example(i, start, last_time, example, dname)


#####
# Natural Questions
#####
        
dname = datasets_evaluation[1]
if dname in task_list:
    dataset = load_dataset(dname, split="validation")
    shuffled = iter(dataset.shuffle(seed=seed))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = example["question"]["text"]

        answer = ""
        for sa in example["annotations"]["short_answers"]:
            if len(sa["text"]) > 0:
                answer = sa["text"][0]
                break

        prediction = answer
        example_id = hashlib.md5((query + "_" + prediction).encode()).hexdigest()
        example = make_example(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="mc", domain="trivia")
        
        save_example(i, start, last_time, example, dname)

#####
# Trivia QA
#####
        
dname = datasets_evaluation[2]
if dname in task_list:
    dataset = load_dataset(dname, 'rc.web.nocontext', split="validation")
    shuffled = iter(dataset.shuffle(seed=seed))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = example["question"]

        answer = example["answer"]["value"]

        prediction = answer
        example_id = hashlib.md5((query + "_" + prediction).encode()).hexdigest()
        example = make_example(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="mc", domain="trivia")
        
        save_example(i, start, last_time, example, dname)

dname = datasets_evaluation[2]
if dname in task_list:
    dataset = load_dataset(dname, 'rc.wikipedia.nocontext', split="validation")
    shuffled = iter(dataset.shuffle(seed=seed))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = example["question"]

        answer = example["answer"]["value"]

        prediction = answer
        example_id = hashlib.md5((query + "_" + prediction).encode()).hexdigest()
        example = make_example(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="mc", domain="trivia")
        
        save_example(i, start, last_time, example, dname)

#####
# Hotpot QA
#####
        
dname = datasets_evaluation[3]
if dname in task_list:
    dataset = load_dataset(dname, 'distractor', split="validation")
    shuffled = iter(dataset.shuffle(seed=seed))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = example["question"]

        answer = example["answer"]

        prediction = answer
        example_id = hashlib.md5((query + "_" + prediction).encode()).hexdigest()
        example = make_example(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="mc", domain="trivia")
        
        save_example(i, start, last_time, example, dname)
# Save the dataset to a file
save_examples(i, start, last_time, [], dname, force=True)