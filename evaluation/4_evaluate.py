from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_similarity,
)
from dotenv import load_dotenv
from pandas import read_parquet

load_dotenv("../.env")

###########
# Config variables
reranker = "llmware"
llm = "llmware"
###########

# Load the dataset
data_path = f"data/dataset/itrf_evaluation_{reranker}_{llm}"
itrf = read_parquet(f"{data_path}.parquet")

def run_eval(sample):
    # TODO VERIFY THIS

    # Select answer by reranker times llm score

    sample["faithfulness"] = faithfulness(sample["answer"], sample["contexts"])
    sample["answer_relevancy"] = answer_relevancy(sample["answer"], sample["contexts"])

    # Calculate the context recall and precision
    sample["context_recall"] = context_recall(sample["answer"], sample["contexts"])
    sample["context_precision"] = context_precision(sample["answer"], sample["contexts"])

    pass

itrf = itrf.apply(run_eval, axis=1)
itrf.to_parquet(f"{data_path}_evaluated.parquet")