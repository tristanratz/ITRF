# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(path):
    # Set up the model and tokenizer
    model_name =  path
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def basic_inference(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")

    generate_ids = model.generate(inputs.input_ids, max_length=30)
    answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return answer