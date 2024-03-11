from datasets import load_dataset, Dataset
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import models, QdrantClient
from langchain.vectorstores import Qdrant
import time
from datetime import timedelta
import pandas as pd
from pandas import DataFrame
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=7500, help='The number of samples per dataset')
parser.add_argument('--collection', type=str, default="retriever", help='Collection name in the vector database')
parser.add_argument('--k', type=int, default=3, help='The number of contexts per sample')
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
data_path = '../data/dataset/itrf_dataset_llm'
#######

itrf_dataset_buffer = []

datasets_openqa = ["tau/commonsense_qa", "math_qa", "web_questions", "wiki_qa", "yahoo_answers_qa", "freebase_qa", "ms_marco"]
datasets_reading = ["coqa", "drop", "narrativeqa", "newsqa", ("pubmed_qa", "pqa_unlabeled"), "quail", "quarel", "squad_v2"] # "natural_questions", "trivia_qa",  "search_qa", "", "duorc", "ropes"
datasets_summ = ["cnn_dailymail"]
datasets_cot = ["aqua_rat", "yangdong/ecqa", "gsm8k", "hendrycks/competition_math", "metaeval/strategy-qa"]

task_list = []
task_list.extend(datasets_openqa)
task_list.extend(datasets_reading)
task_list.extend(datasets_summ)
task_list.extend(datasets_cot)

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

embedding = HuggingFaceBgeEmbeddings(model_name="../models/retriever/bge-base-en-v1.5", model_kwargs={"device": "cuda:0"})

# Create the retriever
client = QdrantClient(url="http://localhost:6333")
db = Qdrant(client, 
            collection_name=col_name,
            embeddings=embedding,
            )

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
def make_example(query: str, prediction:str, dataset_name:str, context = None, example_id = None, k: int = 3, split = "llm", retrieval = True, task="", domain=""):
    contexts = []
    if retrieval:
        # Search for the query
        results = search(query, k=k)

        # Get the text of the results
        contexts = [{
            "text": result[0].page_content, 
            "src": result[0].metadata["src"] if "src" in result[0].metadata.keys() else "unknown", 
            "id": result[0].metadata["id"] if "id" in result[0].metadata.keys() else result[0].metadata["title"],
            "original_context": False } 
            for result in results]
    
    # Add the original context
    if context:
        contexts.append({
            "text": context, 
            "src": dataset_name, 
            "id": str(example_id), 
            "original_context": True
            })

    return { 
        "split": split, 
        "query": query, 
        "prediction": prediction, 
        "contexts": contexts, 
        "src": dataset_name, 
        "id": str(example_id),
        "task": task,
        "domain": domain, 
        }

# Depending if the examples should be searved consecutively or not (shuffling not possible with all contexts in one example)
# This function creates a list of examples with one for each context
def make_examples(query: str, prediction:str, dataset_name:str, context = None, example_id = None, k: int = 3, retrieval = True, task="", domain=""):
    examples = []
    ex = make_example(query, prediction, dataset_name, context, example_id, k, retrieval=retrieval, task=task, domain=domain)
    for c in ex["contexts"]:
        examples.append({ 
            "split": ex["split"], 
            "query": ex["query"].strip(), 
            "prediction": ex["prediction"].strip(), 
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
    shuffled = iter(dataset.shuffle(seed=2024))
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
    shuffled = iter(dataset.shuffle(seed=2024))
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
    shuffled = iter(dataset.shuffle(seed=2024))
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
    shuffled = iter(dataset.shuffle(seed=2024))
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
    shuffled = iter(dataset.shuffle(seed=2024))
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

##################
# Reading Comprehension
##################

#####
# CoQA
#####

dname = datasets_reading[0]
if dname in task_list:
    dataset = load_dataset(dname, split="train")
    shuffled = iter(dataset.shuffle(seed=2024))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = example["questions"][0]
        prediction =  example["answers"]['input_text'][0]
        example_id = example["source"] + "_" + str(i)
        context = example["story"]
        examples = make_examples(query, prediction, dname, example_id=example_id, context=context, k=k, retrieval=True, task="qa", domain="rc")
            
        save_examples(i, start, last_time, examples, dname)

#####
# drop
#####

dname = datasets_reading[1]
if dname in task_list:
    dataset = load_dataset(dname, split="train").to_pandas()
    dataset.drop_duplicates(subset=["section_id"], keep="first", inplace=True)
    dataset = Dataset.from_pandas(dataset)
    shuffled = iter(dataset.shuffle(seed=2024))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = example["question"]
        spans = example["answers_spans"]['spans']
        if len(spans) > 1:
            prediction =  ", ".join(spans)
        else:
            prediction = spans[0]
        
        example_id = example["query_id"]
        context = example["passage"]
        examples = make_examples(query, prediction, dname, example_id=example_id, context=context, k=k, retrieval=True, task="qa", domain="rc")
            
        save_examples(i, start, last_time, examples, dname)

#####
# narrativeqa
#####

dname = datasets_reading[2]
if dname in task_list:
    dataset = load_dataset(dname, split="train")
    shuffled = iter(dataset.shuffle(seed=2024))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = example["question"]["text"]
        prediction =  example["answers"][0]['text']
        example_id = example["document"]["id"] + "_" + str(i)
        context = example["document"]["summary"]["text"]
        examples = make_examples(query, prediction, dname, example_id=example_id, context=context, k=k, retrieval=True, task="qa", domain="rc")
            
        save_examples(i, start, last_time, examples, dname)

#####
# pubmedqa
#####

dname = datasets_reading[4][0]
if dname in task_list:
    dataset = load_dataset(dname,  datasets_reading[4][1], split="train")
    shuffled = iter(dataset.shuffle(seed=2024))
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
# quail
#####

dname = datasets_reading[5]
if dname in task_list:
    dataset = load_dataset(dname, split="train")
    shuffled = iter(dataset.shuffle(seed=2024))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = example["question"]
        prediction =  example["answers"][example["correct_answer_id"]]
        if example["question_type"] == "Unanswerable":
            prediction = "I don't know."
        example_id = example["id"]
        context = " ".join(example["context"])
        examples = make_examples(query, prediction, dname, example_id=example_id, context=context, k=k, retrieval=True, task="qa", domain="rc")
            
        save_examples(i, start, last_time, examples, dname)

#####
# quarel
#####

dname = datasets_reading[6]
if dname in task_list:
    dataset = load_dataset(dname, split="train")
    shuffled = iter(dataset.shuffle(seed=2024))
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


#####
# squad_v2
#####

dname = datasets_reading[7]
if dname in task_list:
    dataset = load_dataset(dname, split="train")
    shuffled = iter(dataset.shuffle(seed=2024))
    start = time.time()
    last_time = start
    for i in range(n*2):
        example = next(shuffled, sentinel)
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = example["question"]
        answer = example["answers"]["text"]
        prediction = ""
        if len(answer) == 0:
            prediction = "I don't know."
        else:
            prediction = answer[0]
        example_id = example["id"]
        context = " ".join(example["context"])
        examples = make_examples(query, prediction, dname, example_id=example_id, context=context, k=k, retrieval=True, task="qa", domain="rc")
            
        save_examples(i, start, last_time, examples, dname)

##################
# Summarization
##################

#####
# cnn_dailymail
#####

dname = "cnn_dailymail"
if dname in task_list:
    dataset = load_dataset(dname, "1.0.0", split="train")
    shuffled = iter(dataset.shuffle(seed=2024))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = "Summarize this article"
        prediction =  example["highlights"]
        example_id = example["id"]
        examples = make_examples(query, prediction, dname, example_id=example_id, k=k, context=example["article"], retrieval=False, task="sum", domain="sum")
            
        save_examples(i, start, last_time, examples, dname)

##################
# Chain of thought reasoning
##################

#####
# aqua_rat
#####

dname = datasets_cot[0]
if dname in task_list:
    dataset = load_dataset(dname, "raw", split="train")
    shuffled = iter(dataset.shuffle(seed=2024))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = example["question"] + "\n" + example["rationale"]

        options = "\nOptions: "
        correct = -1
        for idx, o in enumerate(example["options"]):
            options += f"\n{o}"
            if o[0] == example["correct"]:
                correct = idx

        if options_enabled:
            query += options
            prediction = example["options"][correct]   
        else:
            prediction = example["options"][correct].split(")")[1]

        example_id = dname + "_" + str(i)
        examples = make_examples(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="mc", domain="cotr")
            
        save_examples(i, start, last_time, examples, dname)
#####
# yangdong/ecqa
#####

dname = datasets_cot[1]
if dname in task_list:
    dataset = load_dataset(dname, split="train")
    shuffled = iter(dataset.shuffle(seed=2024))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = example["q_text"]
        query += "\n" + example["taskA_pos"]

        options = "\nOptions: "
        correct = -1
        for idx in range(5):
            options += f"{idx+1}) {example[f'q_op{idx+1}']}"
            if example["q_ans"] == example[f'q_op{idx+1}']:
                correct = idx + 1
        if options_enabled:
            query += options
            prediction = f"{correct}) {example[f'q_op{correct}']}"
        else:
            prediction = example[f'q_op{correct}']
        
        example_id = example["q_no"]
        examples = make_examples(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="mc", domain="cotr")
            
        save_examples(i, start, last_time, examples, dname)

#####
# gsm8k
#####

dname = datasets_cot[2]
if dname in task_list:
    dataset = load_dataset(dname, "main", split="train", streaming=True)
    shuffled = iter(dataset.shuffle(buffer_size=10_000, seed=2024))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        rational = example["answer"].split("####")
        query = example["question"] + "\n" + rational[0]
        prediction =  rational[1]
        example_id = dname + "_" + str(i)
        examples = make_examples(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="qa", domain="cotr")
            
        save_examples(i, start, last_time, examples, dname)

#####
# MATH -> hendrycks/competition_math
#####
    
dname = datasets_cot[3]
if dname in task_list:
    dataset = load_dataset(dname, split="train")
    shuffled = iter(dataset.shuffle(seed=2024))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = example["problem"]
        prediction =  example["solution"]
        example_id = dname + "_" + str(i)
        examples = make_examples(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="qa", domain="cotr")
            
        save_examples(i, start, last_time, examples, dname)


#####
# metaeval/strategy-qa
#####
    
dname = datasets_cot[4]
if dname in task_list:
    dataset = load_dataset(dname, split="train")
    shuffled = iter(dataset.shuffle(seed=2024))
    start = time.time()
    last_time = start
    sample_range = skip(shuffled, dname)
    for i in sample_range:
        example = next(shuffled, sentinel)
        if example is sentinel:
            print(f"Dataset {dname} run out of examples after {i} examples")
            break

        query = example["question"]
        rational = ""
        for idx, de in enumerate(example["decomposition"]):
            if len(example["facts"]) > idx:
                rational += f"{de} {example['facts']} "
            else:
                rational += f"{de}"
        query += "\n" + rational
        prediction =  "True" if example["answer"] else "False"
        example_id = example["qid"]
        examples = make_examples(query, prediction, dname, example_id=example_id, k=k, retrieval=True, task="tf", domain="cotr")
            
        save_examples(i, start, last_time, examples, dname)
# Force save after the last example
save_examples(0, start, last_time, [], dname, force=True)
