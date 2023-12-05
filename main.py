from model import load_model, basic_inference


def main():
    model, tokenizer = load_model("../llama/llama-2-13b/") # consolidated.00.pth
    while True:
        inp = input(">")
        answer = basic_inference(inp, model, tokenizer)
        print("->", answer)

if __name__ == "__main__":
    main()
