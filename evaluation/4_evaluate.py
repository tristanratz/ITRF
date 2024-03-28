from dotenv import load_dotenv
load_dotenv("../.env")

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_relevancy,
    answer_similarity,
)
from ragas import evaluate
from pandas import read_json, read_parquet, DataFrame
import pandas as pd
from datasets import Dataset
from argparse import ArgumentParser
import json

parser = ArgumentParser()

# Parse the arguments
parser.add_argument("--reranker", type=str, help="The reranker model used", required=True)
parser.add_argument("--llm", type=str, help="The llm model used", required=True)
parser.add_argument("--n", type=int, default=800, help="The number of samples to evaluate")
parser.add_argument("--resume", type=bool, default=False, help="Continue from the last evaluation")

args = parser.parse_args()


###########
# Config variables
reranker = args.reranker
llm = args.llm
seed = 4048
n = args.n
###########

# Load the dataset
input_path = f"../data/dataset/itrf_evaluation_generation_{reranker}_{llm}_mse"
itrf = read_parquet(f"{input_path}.parquet")
data_path =  f"./evaluation_results/itrf_evaluation_{reranker}_{llm}"
datasets = itrf["src"].unique()

def rank_context(sample, get_answer=False):
    # Select answer by reranker times llm score
    # Get context with the highest score = llm_softmax * reranker_softmax
    contexts = sample["all_contexts"]
    highest_score_context = max(contexts, key=lambda x: x['reranker_softmax'] * x['llm_softmax'])

    if get_answer:
        return highest_score_context["predicted"].strip()
    else:
        return [highest_score_context["text"]]

# Prepare data
itrf["question"] = itrf["query"]
itrf.rename(columns={"contexts": "all_contexts",}, inplace=True)
itrf["answer"] = itrf.apply(rank_context, axis=1, get_answer=True)
itrf["ground_truths"] = itrf["ground_truth"].apply(lambda x: [x])  # "ground_truth": "ground_truths"
itrf["contexts"] = itrf.apply(rank_context, axis=1)

result = None
result_score = []

if args.resume:
    result = read_json(f"{data_path}_evaluated.json")
    result_score = json.load(open(f"{data_path}_evaluated_scores.json"))

for d in datasets:

    if args.resume and d in result["src"].unique():
        continue

    print(f"Processing dataset: {d}")
    df = itrf[itrf["src"] == d].sample(n, random_state=seed)
    ds = Dataset.from_pandas(df)
    results = evaluate(ds, metrics=[
        faithfulness, 
        answer_relevancy, 
        answer_similarity, 
        context_recall, 
        context_relevancy
    ])
    if result is None:
        result = results.to_pandas()
    else:
        result = pd.concat([result, results.to_pandas()])

    results["dataset"] = d
    result_score.append(results)

    json.dump(result_score, open(f"{data_path}_evaluated_scores.json", "w"))

    result.reset_index(inplace=True, drop=True)
    result.to_json(f"{data_path}_evaluated.json")