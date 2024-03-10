from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
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
    base_model = "LoftQ/Llama-2-13b-hf-4bit-64rank"
    output_dir = "../models/itrf/13b-loftq"


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
        logging_steps=1,
        optim="adamw_bnb_8bit", # Adafactor can be less memory intensive than AdamW but may require more steps to converge, adamw_torch
        logging_dir="../data/training_output",
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1,
        save_steps=100,
        save_total_limit=10,
        load_best_model_at_end=True,
    )

    # Load the model
    if zero_stage > 2:
        # Getting error with Zero stage 3 with deepspeed.zero.Init() I think it's because of the flash attention – disable for stage 3
        
        model = AutoModelForCausalLM.from_pretrained(
                            base_model, 
                            token=token,
                            attn_implementation="flash_attention_2",
                            # torch_dtype=torch.bfloat16,  # you may change it with different models
                            load_in_4bit=True,
                            # ignore_mismatched_sizes=True,
                            quantization_config=BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is recommended
                                bnb_4bit_use_double_quant=False,
                                bnb_4bit_quant_type='nf4',
                            ),
                        )
        peft_model = PeftModel.from_pretrained(
            model,
            base_model,
            subfolder="loftq_init",
            is_trainable=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=token, torch_dtype=torch.bfloat16, return_tensors="pt")
    else:
        model = AutoModelForCausalLM.from_pretrained(
                            base_model, 
                            device_map=device,
                            token=token,
                            attn_implementation="flash_attention_2",
                            load_in_4bit=True,
                            quantization_config=BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is recommended
                                bnb_4bit_use_double_quant=False,
                                bnb_4bit_quant_type='nf4',
                            ),
                        )
        peft_model = PeftModel.from_pretrained(
            model,
            base_model,
            subfolder="loftq_init",
            is_trainable=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=token, return_tensors="pt", device_map=device)
    
    # https://github.com/huggingface/tokenizers/issues/247
    tokenizer.add_special_tokens({"additional_special_tokens": ["[INST]", "[/INST]", "[ANS]", "[/ANS]"]})
    peft_model.resize_token_embeddings(len(tokenizer))
    
    # For batch tokenization and packing
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
    
    peft_model.enable_input_require_grads()
    peft_model.gradient_checkpointing_enable()
    prepare_model_for_kbit_training(peft_model)
    accelerator.print("Model created")

    # Load the dataset
    dataset = load_dataset("parquet", data_files="../data/dataset/rarit_dataset_llm_processed2.parquet")

    # split thte data to train/val set
    val_set_size = 0.1
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
    
    
    # Train the model
    accelerator.print("Training started")
    trainer.train()
    accelerator.print("Training finished")

    
    # Save the model
    if accelerator.is_main_process:    
        # peft_model.save_pretrained(f"{output_dir}/peft_model")
        trainer.save_model(f"{output_dir}/trainer")
        accelerator.print("Model saved")


if __name__ == "__main__":
    main()