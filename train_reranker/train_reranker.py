from datasets import load_dataset
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from sentence_transformers import InputExample
from dotenv import load_dotenv
import os
from sentence_transformers import  CrossEncoder
from torch.utils.data import DataLoader
from nvidia_smi import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from eval import MSEEval
# Cross Encoder BGE

device = "cuda:1"

val_set_size = 0.05

base_model = "BAAI/bge-reranker-large"
dataset = load_dataset("parquet", data_files="../data/dataset/itrf_dataset_reranker_processed.parquet")
output_dir = "../models/itrf_reranker-large"
len(dataset["train"])
# open("../data/dataset/itrf_dataset_llm.parquet")

# split the data to train/val set
train_val = dataset["train"].train_test_split(
    test_size=val_set_size, shuffle=True, seed=2024
)
train_data = train_val["train"].shuffle(seed=2024)
val_data = train_val["test"].shuffle(seed=2024)

# Create Input Samples
print("Create Input Samples")
train_samples = [InputExample(texts=[ex["query"], ex["context"]], label=ex["llm_weighted_softmax"]) for ex in train_data]
val_samples = [InputExample(texts=[ex["query"], ex["context"]], label=ex["llm_weighted_softmax"]) for ex in val_data]

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
val_dataloader = DataLoader(val_samples, shuffle=True, batch_size=3)
evaluator = MSEEval(val_dataloader, )

# Load the model
print("Load the model")
cross_encoder = CrossEncoder(base_model, num_labels=1, device=device)

# Train the model
print("Train the model")
cross_encoder.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=1,
    warmup_steps=100,
    evaluation_steps=1000,
    output_path=output_dir,
    save_best_model=True,
    use_amp=True,
    scheduler= 'warmupcosine',
    show_progress_bar=True,
)

# Save the model
print("Save the model")
cross_encoder.save(f"{output_dir}/final_model")