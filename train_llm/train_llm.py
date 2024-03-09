from datasets import load_dataset
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments
from peft import get_peft_model, LoftQConfig, LoraConfig, PeftModel, PeftConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from dotenv import load_dotenv
from accelerate import Accelerator
import os
from nvidia_smi import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import deepspeed

# Load environment variables from .env file
load_dotenv("../.env")


# Access the environment variable
token = os.getenv('HUGGINGFACE_TOKEN')
accelerator = Accelerator()

device = accelerator.device
zero_stage = 3

accelerator.print(f"Using device: {device}")

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    accelerator.print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    accelerator.print(f"Time: {result.metrics['train_runtime']:.2f}")
    accelerator.print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

# Bring the examples in the form of the prompt
def _formatting_func(example):
    text = f"<s>Background: {example['context']}\n\n[INST]{example['query']}[/INST][ANS]{example['prediction']}</s>"
    return text

def main():
    max_seq_length = 2048 # Supports automatic RoPE Scaling, so choose any number

    packing = True # Packing multiple examples into one sequence

    base_model = "meta-llama/Llama-2-7b-chat-hf"

    output_dir = "../models/itrf/7b-lora"

    # Set the training arguments
    args = TrainingArguments(
        output_dir = output_dir,
        # auto_find_batch_size=True,
        per_device_train_batch_size=2, # Batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus (if using DataParallel)
        per_device_eval_batch_size=2, # Batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus (if using DataParallel)
        gradient_accumulation_steps=8, # Helps to fit larger batch sizes into memory, slightly increases training time
        eval_accumulation_steps=8,
        gradient_checkpointing=True,
        warmup_steps=100,
        # num_train_epochs=500,
        max_steps=1000,
        learning_rate=1e-5,
        bf16=True, # Use bfloat16 precision if u have the right hardware – 
        logging_steps=10,
        optim="adamw_bnb_8bit", # Adafactor can be less memory intensive than AdamW but may require more steps to converge, adamw_torch
        logging_dir="../data/training_output",
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=10,
        load_best_model_at_end=True,
    )

    # Load the model
    if zero_stage > 2:
        # Getting error with Zero stage 3 with deepspeed.init.Init() I think it's because of the flash attention – disable for stage 3
        # with deepspeed.zero.Init():
        model = LlamaForCausalLM.from_pretrained(
            base_model, 
            # load_in_4bit=True,
            attn_implementation="flash_attention_2",
            torch_dtype="auto", # torch.bfloat16,  # you may change it with different models
            token=token)
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=token, torch_dtype="auto", return_tensors="pt")
    else:
        model = LlamaForCausalLM.from_pretrained(base_model, attn_implementation="flash_attention_2",  device_map=device, token=token)
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=token, return_tensors="pt", device_map=device)
    
    # https://github.com/huggingface/tokenizers/issues/247
    tokenizer.add_special_tokens({"additional_special_tokens": ["[INST]", "[/INST]", "[ANS]", "[/ANS]"]})
    model.resize_token_embeddings(len(tokenizer))
    
    # For batch tokenization and packing
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    # Set the peft arguments
    # loftq_config = LoftQConfig(loftq_bits=4) # set 8bit quantization
    lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            # init_lora_weights="loftq", 
            # loftq_config=loftq_config,
            task_type="CAUSAL_LM",
        )
    
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    prepare_model_for_kbit_training(model)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    accelerator.print("Model created")

    # Load the dataset
    val_set_size = 0.1

    dataset = load_dataset("parquet", data_files="../data/dataset/rarit_dataset_llm_processed2.parquet")
    len(dataset["train"])
    # open("../data/dataset/rarit_dataset_llm.parquet")

    #split thte data to train/val set
    train_val = dataset["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=2024
    )
    train_data = train_val["train"].shuffle(seed=2024)
    val_data = train_val["test"].shuffle(seed=2024)

    accelerator.print("Data loaded")

    trainer = accelerator.prepare(SFTTrainer(
            model=peft_model,
            args=args,
            train_dataset=train_data,
            eval_dataset=val_data, 
            packing=packing, # Packing multiple examples into one sequence
            formatting_func=_formatting_func,
            # data_collator=data_collator,
            max_seq_length=max_seq_length,
        ))
    
    accelerator.print("Training started")
    
    # Train the model
    trainer.train(resume_from_checkpoint = True)

    accelerator.print("Training finished")
    
    if accelerator.is_main_process:
        # Save the model
        # peft_model.save_pretrained(f"{output_dir}/peft_model")
        trainer.save_model(f"{output_dir}/trainer")

        accelerator.print("Model saved")

if __name__ == "__main__":
    main()