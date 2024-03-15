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

import sys
import os
sys.path.append(os.path.abspath('../'))

from model.llm import LLM

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=3500, help='The number of samples per dataset')
parser.add_argument('--collection', type=str, default="retriever", help='Collection name in the vector database')
parser.add_argument('--k', type=int, default=10, help='The number of contexts per sample')
parser.add_argument('--options', type=bool, default=False, help='If the multiple choice options should be part of query')
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
data_path = '../data/dataset/itrf_dataset_reranker'
#######

datasets_openqa = ["tau/commonsense_qa", "math_qa", "web_questions", "wiki_qa", "yahoo_answers_qa", "freebase_qa",("ms_marco", 'v2.1')]
datasets_reading = [("pubmed_qa", "pqa_unlabeled"), "quarel", ]

task_list = []
task_list.extend(datasets_openqa)
task_list.extend(datasets_reading)

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


llm = LLM(size=7, quantized=False)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum()

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

# This function creates an example with the query and the prediction and the top k results
def make_example(query: str, ground_truth:str, dataset_name:str, context = None, example_id = None, k: int = 3, split = "llm", retrieval = True, task="", domain=""):
    contexts = []
    if retrieval:
        # Search for the query
        results = search(query, k=k)

        # Get the softmax of the scores
        retriever_softmax = softmax([result[1] for result in results])
        context_texts = [llm.format_prompt(query, result[0].page_content) for result in results]
        llm_scores = llm.to_tokens_and_logprobs(context_texts, [ground_truth] * k)[1]
        llm_softmax = softmax(llm_scores)

        # Get the text of the results
        contexts = [{
            "text": result[0].page_content, 
            "src": result[0].metadata["src"] if "src" in result[0].metadata.keys() else "unknown", 
            "id": str(result[0].metadata["id"] if "id" in result[0].metadata.keys() else result[0].metadata["title"]),
            "retriever_score": result[1],
            "llm_score": lscore,
            "retriever_softmax": rsoft, 
            "llm_softmax": lsoft, 
            "llm_weighted_softmax": rsoft * lsoft, 
            "original_context": False } 
            for result, rsoft, lsoft, lscore in zip(results, retriever_softmax, llm_softmax, llm_scores)]
    
    # Add the original context
    # if context:
    #     contexts.append({
    #         "text": context, 
    #         "src": dataset_name, 
    #         "id": str(example_id), 
    #         "original_context": True
    #         })

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

def make_examples(query: str, prediction:str, dataset_name:str, context = None, example_id = None, k: int = 3, retrieval = True, task="", domain=""):
    examples = []
    ex = make_example(query, prediction, dataset_name, context, example_id, k, retrieval=retrieval, task=task, domain=domain)
    for c in ex["contexts"]:
        examples.append({ 
            "split": ex["split"], 
            "query": ex["query"], 
            "ground_truth": ex["ground_truth"], 

            "retriever_score": c["retriever_score"],
            "llm_score": c["llm_score"],
            "retriever_softmax": c["retriever_softmax"], 
            "llm_softmax": c["llm_softmax"], 
            "llm_weighted_softmax": c["llm_weighted_softmax"], 

            "context": c["text"], 
            "src": ex["src"], 
            "id": str(ex["id"]), 
            "context_src": c["src"], 
            "context_id": str(c["id"]), 
            "original_context": c["original_context"],
            "task": task,
            "domain": domain,
            })
    return examples

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
        itrf.to_parquet(f"{data_path}.parquet")
        # itrf.to_csv(f"{data_path}.csv", escapechar='\\')

def skip(dataset, dname):
    if dname == continue_point:
        for _ in range(args.sample_skip+1):
            next(dataset)
        print("Continue from", args.sample_skip+1, "to", n)
        return range(args.sample_skip+1, n)
    return range(n)



##################
# OpenQA
##################

#####
# TriviaQA
#####

dname = datasets_openqa[0]
if dname in task_list:
    dataset = load_dataset(dname, split="train")
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
            options = "\n Options: "
        answer = -1
        for idx, o in enumerate(example["choices"]["label"]):
            if options_enabled:
                options += f"{o}: {example['choices']['text'][idx]}, "
            if example["answerKey"] == o:
                answer = idx
        if options_enabled: 
            options += "\n"

        query = example["question"] + options
        if options_enabled:
            prediction = f"{example['answerKey']}) {example['choices']['text'][answer]}"
        else:
            prediction =  f"{example['choices']['text'][answer]}"
        example_id = example["id"] + "_" + example["question_concept"]
        examples = make_examples(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="mc", domain="openqa")
        
        save_examples(i, start, last_time, examples, dname)

#####
# MathQA
#####
dname = datasets_openqa[1]
if dname in task_list:
    dataset = load_dataset(dname, split="train")
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
            options =  "\nOptions: " + example["options"]
        query = example["Problem"] + options

        answer = ""
        if options_enabled:
            answer = example["correct"] + ") "

        # Extract actual answer from options
        for idx, o in enumerate(example["options"].split(",")):
            a = o.split(" ) ")
            if example["correct"] == a[0].strip():
                answer += a[1]

        prediction = answer
        example_id = example["category"] + "_" + str(i)
        examples = make_examples(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="mc", domain="openqa")
        
        save_examples(i, start, last_time, examples, dname)

#####
# web_questions
#####

dname = datasets_openqa[2]
if dname in task_list:
    dataset = load_dataset(dname, split="train")
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
        prediction =  example["answers"][0]
        example_id = example["url"] + "_" + example["answers"][0]
        examples = make_examples(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="qa", domain="openqa")
        
        save_examples(i, start, last_time, examples, dname)

#####
# WikiQA
#####

dname = datasets_openqa[3]
if dname in task_list:
    dataset = load_dataset(dname, split="train")
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
        prediction =  example["answer"]
        example_id = example["question_id"] + "_" + example["document_title"]
        examples = make_examples(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="qa", domain="openqa")
            
        save_examples(i, start, last_time, examples, dname)

#####
# yahoo_answers_qa
#####

dname = datasets_openqa[4]
if dname in task_list:
    dataset = load_dataset(dname, split="train")
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
        prediction =  example["answer"]
        example_id = example["id"] + "_" + example["main_category"]
        examples = make_examples(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="qa", domain="openqa")
            
        save_examples(i, start, last_time, examples, dname)

#####
# freebase_qa
#####
        
dname = datasets_openqa[5]
if dname in task_list:
    dataset = load_dataset(dname, split="train")
    shuffled = iter(dataset.shuffle(seed=seed))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = example["RawQuestion"]
        prediction =  example['Parses']["Answers"][0]["AnswersName"][0][0]
        example_id = example["Question-ID"]
        examples = make_examples(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="qa", domain="openqa")
        
        save_examples(i, start, last_time, examples, dname)

#####
# ms_marco
#####
        
dname = datasets_openqa[6][0]
if datasets_openqa[6] in task_list:
    dataset = load_dataset(dname, datasets_openqa[6][1], split="train")
    shuffled = iter(dataset.shuffle(seed=seed))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = example["query"]
        example_id = dname + example["query_type"] + str(example["query_id"])

        if 1 in example["passages"]["is_selected"]:
            prediction =  example['answers'][0]
            context_id = example["passages"]["is_selected"].index(1)
            context = example["passages"]["passage_text"][context_id]
        else:
            prediction = "I don't know."
            context = example["passages"]["passage_text"][0]

        examples = make_examples(query, prediction, dname, example_id=example_id, context=context, k=k, retrieval=True, task="qa", domain="openqa")
        
        save_examples(i, start, last_time, examples, dname)


##################
# Reading Comprehension
##################
        
#####
# pubmedqa
#####

dname = datasets_reading[0][0]
if datasets_reading[0] in task_list:
    dataset = load_dataset(dname,  datasets_reading[0][1], split="train")
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
        prediction =  example["long_answer"]
        example_id = example["pubid"]
        context = " ".join(example["context"]["contexts"])
        examples = make_examples(query, prediction, dname, example_id=example_id, context=context, k=k, retrieval=True, task="qa", domain="rc")
            
        save_examples(i, start, last_time, examples, dname)

#####
# quarel
#####

dname = datasets_reading[1]
if dname in task_list:
    dataset = load_dataset(dname, split="train")
    shuffled = iter(dataset.shuffle(seed=seed))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = example["question"].split("(A)")[0]
        if options_enabled:
            query += "\n Options: (A)" + example["question"].split("(A)")[1]

        answers = example["question"].split("(A)")[1].split("(B)")
        prediction =  answers[example["answer_index"]].strip()
        example_id = example["id"]
        examples = make_examples(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="mc", domain="rc")
            
        save_examples(i, start, last_time, examples, dname)