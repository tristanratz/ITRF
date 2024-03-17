import torch
import os
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv
from accelerate import Accelerator
import numpy as np


class LLM:
    def __init__(self, size="13", quantized=True, model_path=None, adapter=True):
        # Load environment variables from .env file
        load_dotenv("../.env")

        # Access the environment variable
        token = os.getenv('HUGGINGFACE_TOKEN')
        
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.accelerator.print(f"Using device: {self.device}")

        base_model = f"meta-llama/Llama-2-{size}b-chat-hf"
        load_dotenv("../.env")
        if model_path is not None:
            base_model = model_path

        if adapter:
            adapter_path = f"tristanratz/itrf-{size}b-{'q' if quantized else ''}lora"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, token=token, torch_dtype="auto", return_tensors="pt")

        if quantized:
            self.model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    device_map=self.device,
                    attn_implementation="flash_attention_2",
                    torch_dtype="auto", # torch.bfloat16,  # you may change it with different models
                    quantization_config=BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is recommended
                                    bnb_4bit_use_double_quant=False,
                                    bnb_4bit_quant_type='nf4',
                                ),
                    token=token)
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["[INST]", "[/INST]", "[ANS]", "[/ANS]"]})
        else:
            self.model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    device_map=self.device,
                    attn_implementation="flash_attention_2",
                    torch_dtype="auto", # torch.bfloat16,  # you may change it with different models
                    token=token)
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<>", "<inst_e>"]})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

        # For batch tokenization and packing
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.padding_side = "left"

        if adapter:
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
            )


    def inference(self, input):
        return self.inference_score(input)[0]

    def inference_score(self, input):

        outputs, tok_scores, probs = self.inference_batch_score([input])

        return outputs[0], probs[0]
    
    def inference_batch(self, inputs):
        return self.inference_batch_score(inputs)[0]
    
    def inference_batch_score(self, inputs):
        """
        Generate the output sequences for the given input sequences.
        Additionally return the log-probabilities of the tokens and the probability of the whole sequence.

        Args:
            inputs (List[str]): The input sequences.
        """

        model_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        input_length = model_inputs.input_ids.shape[1]

        outputs = self.model.generate(
            **model_inputs,
            output_scores=True, 
            do_sample=True,
            temperature=0.7,
            return_dict_in_generate=True,
        ) # do_sample=True to generate text more creatively

        # Decode the output sequences
        output_seqs = self.tokenizer.batch_decode(outputs.sequences[:, input_length:], skip_special_tokens=True)
        # whole_seqs = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        # Calculate the log-probabilities of the tokens
        scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True,
        ).cpu().numpy()

        tok_scores = []
        for sc, out in zip(scores, outputs.sequences[:, input_length:]):
            tsc = []
            for s, t in zip(sc, out):
                if not t in self.tokenizer.all_special_ids:
                    tsc.append((self.tokenizer.decode(t)[0], np.exp(s)))
            tok_scores.append(tsc)

        # Calculate the probability of the whole sequence
        output_length = model_inputs["input_ids"].shape[1] + np.sum(scores < 0, axis=1)
        length_penalty = self.model.generation_config.length_penalty
        probabilities = np.exp(scores.sum(axis=1) / (output_length**length_penalty))

        return output_seqs, tok_scores, probabilities
    
    def to_tokens_and_logprobs(self, input_texts, pred_texts):
        """
        Returns the tokens and their log-probabilities for the given input and predicted texts.
        As well as the probability of the whole sequence. In other words p(y | x)
        
        Args:
            input_texts (List[str]): The input texts.
            pred_texts (List[str]): The predicted texts.
        """

        # Inspired by: https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/7
        input_ids = self.tokenizer([inp + pred for inp, pred in zip(input_texts, pred_texts)], padding=True, return_tensors="pt").input_ids.to(self.device)
        ans_len = [self.tokenizer([pred], return_tensors="pt").input_ids.shape[1]-1 for pred in pred_texts]
        with torch.no_grad():
            outputs = self.model(input_ids)
        probs = torch.log_softmax(outputs.logits, dim=-1).detach()

        # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
        probs = probs[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
        input_length = [input_id.shape[0] for input_id in input_ids]

        batch = []
        for input_sentence, input_probs, alen, ilen in zip(input_ids, gen_probs, ans_len, input_length):
            text_sequence = []
            i = 0
            for token, p in zip(input_sentence, input_probs):
                i += 1
                if token not in self.tokenizer.all_special_ids:
                    if i > ilen-alen:
                        text_sequence.append((self.tokenizer.decode(token), np.exp(p.item())))
            batch.append(text_sequence)
        
        # Multiplicate the log-probabilities of the tokens to get the probability of the whole sequence
        prob = [np.prod([p for _, p in b]) for b in batch]
        return batch, prob

    def format_prompt(self, query, context):
        text = f"<s>Background: {context}\n\n[INST]{query}[/INST][ANS]"
        return text
    
    def format_llmware_prompt(self, query, context):
        text = f"<human>: {context}\n{query}\n<bot>:"
        return text
