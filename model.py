from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from langchain.chat_models import ChatOpenAI
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from datetime import date
import transformers

load_dotenv()

class Model:

    def __init__(self, model="openai", path = "/workspace/MasterThesis/models/llama7b-ch-hf/"):
        self.model_type = model
        if model=="openai":
            self.llm = ChatOpenAI()
            return
        elif model == "llama":
            model_loaded = LlamaForCausalLM.from_pretrained(path, device_map='auto')
            tokenizer = LlamaTokenizer.from_pretrained(path, device_map='auto')
        elif model == "phi":
            model_loaded = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", device_map="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        else:
            model_loaded = AutoModelForCausalLM.from_pretrained(path)
            tokenizer = AutoTokenizer.from_pretrained(path)
        generate_text = transformers.pipeline(
            model=model_loaded, 
            tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            repetition_penalty=1.1,
            temperature=0.1,
            do_sample=True,
            max_length=500,
        )
        self.llm = HuggingFacePipeline(pipeline=generate_text)


    def get_llm(self):
        return self.llm

def basic_inference(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")

    generate_ids = model.generate(inputs.input_ids, max_length=30)
    answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return answer

if __name__ == "__main__":
    m = Model("llama")
    llm = m.get_llm()

    
    prompt = """<s><<SYS>>
            You are a helpful, respectful and honest assistant called Rarag. Always do...

            Date: {date}
            If you are unsure about an answer, truthfully say "I don't know"
            <</SYS>>

            [INST] Remember you are an assistant [/INST] User:"""
    prompt = str.format(prompt, date=date.today())
    
    inp = "Introduce yourself!"

    while inp != "exit":
        for chunk in llm.stream(prompt + inp):
            print(chunk, end="", flush=True)
        print("\n\n")
        inp = input("> ")