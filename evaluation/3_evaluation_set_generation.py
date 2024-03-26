
from sentence_transformers import CrossEncoder
import time
from datetime import timedelta
import pandas as pd
from pandas import DataFrame
import numpy as np
import argparse
import torch

import sys
import os
sys.path.append(os.path.abspath('../'))

from model.llm import LLM

parser = argparse.ArgumentParser()
parser.add_argument('--collection', type=str, default="retriever", help='Collection name in the vector database')
parser.add_argument('--k', type=int, default=5, help='The number of contexts per sample')
parser.add_argument('--sort_by', type=str, default=None, help='The column to sort the contexts by')
parser.add_argument('--model', type=str, default="itrf", help='The model used for reranking')
parser.add_argument('--device', type=str, default="cuda:2", help='The device used for inference')
parser.add_argument('--reranker', type=str, default=None, help='The reranker model used for reranking')


args = parser.parse_args()

#######
# Config variables
col_name = args.collection
k=args.k
seed = 4048
model = args.model
reranker = args.reranker
if reranker is None:
    input_path = '../data/dataset/itrf_evaluation_retrieval_processed'
else:
    input_path = f'../data/dataset/itrf_evaluation_rerank_{reranker}'
reranker_name = reranker if reranker is not None else "retriever"
data_path = f'../data/dataset/itrf_evaluation_generation_{reranker_name}_{model}'
device=args.device
sort_by=args.sort_by
start = time.time()
#######

if model == "base":
    base_model = "meta-llama/Llama-2-7b-chat-hf"
    llm = LLM(size=7, quantized=False, adapter=False, model_path=base_model, device=device, dftype=torch.bfloat16)
elif model == "itrf":
    llm = LLM(size=7, quantized=False, device=device)
elif model == "llmware":
    base_model = "llmware/dragon-llama-7b-v0"
    llm = LLM(size=7, quantized=False, adapter=False, model_path=base_model, device=device)


sentinel = object() # Used to check if the iterators are empty

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return x / np.sum(x)

itrf = pd.read_parquet(f"{input_path}.parquet")

total = 0
total_contexts = 0

def generate_answer(sample):
    global total
    global total_contexts
    # Calculate the rerank score for all the contexts in the itrf dataset
    # Save the rerank score in the dataset

    contexts = sample["contexts"]
    
    if sort_by is not None:
        sort_ind = np.argsort([c[sort_by] for c in contexts])
        contexts = contexts[sort_ind]

    contexts = contexts[-k:]

    if model == "itrf":
        contexts_texts = [llm.format_prompt(sample["query"], c["text"]) for c in contexts]
    elif model == "llmware":
        contexts_texts = [llm.format_llmware_prompt(sample["query"], c["text"]) for c in contexts]
    else:
        contexts_texts = [llm.format_base_prompt(sample["query"], c["text"]) for c in contexts]
    
    # Predict the answers
    answers, _, scores = llm.inference_batch_score(contexts_texts)
    scores_sm = softmax(scores)

    true_scores = llm.to_tokens_and_logprobs(contexts_texts, [sample["ground_truth"]] * k)[1]
    true_scores_sm = softmax(true_scores)


    # Save the reranked contexts in the dataset
    for i, c in enumerate(contexts):
        contexts[i]["predicted"] = answers[i]
        contexts[i]["llm_score"] = scores[i]
        contexts[i]["llm_softmax"] = scores_sm[i]
        contexts[i]["llm_true_score"] = true_scores[i]
        contexts[i]["llm_true_softmax"] = true_scores_sm[i]

    sort = sort_by if sort_by is not None else "retriever_score"
    # Sort the contexts based on the reranker score
    sort_ind = np.argsort([c["llm_score"]*c[sort] for c in contexts])
    contexts = contexts[sort_ind]

    if total % 100 == 0:
        print(f"Processed {total} samples, with {total_contexts} contexts. Time passed: {timedelta(seconds=time.time()-start)}")
    total += 1
    total_contexts += len(contexts)

    return contexts

itrf["contexts"] = itrf.apply(generate_answer, axis=1)

itrf.to_parquet(f"{data_path}.parquet")