from sentence_transformers import CrossEncoder
import time
from datetime import timedelta
import pandas as pd
from pandas import DataFrame
import numpy as np
import argparse

import sys
import os
# sys.path.append(os.path.abspath('../'))

# from model.llm import LLM

parser = argparse.ArgumentParser()
parser.add_argument('--collection', type=str, default="retriever", help='Collection name in the vector database')
# parser.add_argument('--sample_skip', type=int, default=0, help='Conitnue from this point in the dataset.')
parser.add_argument('--model', type=str, default="itrf", help='The model used for reranking')
parser.add_argument('--device', type=str, default="cuda:2", help='The device used for inference')

args = parser.parse_args()

#######
# Config variables
col_name = args.collection
seed = 4048
model = args.model
data_path = f'../data/dataset/itrf_evaluation_rererank_{model}'
device=args.device

if model == "base":
    base_model = "BAAI/bge-reranker-large"
elif model == "itrf":
    base_model = "../models/itrf_reranker-large/final_model"
elif model == "llmware":
    base_model = "../models/itrf_reranker-large-llmware/final_model"
#######

sentinel = object() # Used to check if the iterators are empty

# llm = LLM(size=7, quantized=False, adapter=False, model_path="llmware/dragon-llama-7b-v0")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return x / np.sum(x)


itrf = pd.read_parquet(f"../data/dataset/itrf_evaluation_retrieval_processed.parquet")
reranker = CrossEncoder(base_model, num_labels=1, device=device)

total = 0
total_contexts = 0

def rerank_sample(sample):
    global total
    global total_contexts
    # Calculate the rerank score for all the contexts in the itrf dataset
    # Save the rerank score in the dataset

    contexts = sample["contexts"]
    contexts_texts = [[sample["query"], c["text"]] for c in contexts]

    # Rerank the contexts
    rerank_scores = reranker.predict(contexts_texts)
    rerank_scores_sm = softmax(rerank_scores)

    # Save the reranked contexts in the dataset
    for i, c in enumerate(contexts):
        contexts[i]["reranker_score"] = rerank_scores[i]
        contexts[i]["reranker_softmax"] = rerank_scores_sm[i]

    # Sort the contexts based on the reranker score
    sort_ind = np.argsort([c["reranker_softmax"] for c in contexts])
    contexts = contexts[sort_ind]

    # sample["contexts"] = contexts

    if total % 100 == 0:
        print(f"Processed {total} samples, with {total_contexts} contexts.")
    total += 1
    total_contexts += len(contexts)

    return contexts

itrf["contexts"] = itrf.apply(rerank_sample, axis=1)

itrf.to_parquet(f"{data_path}.parquet")