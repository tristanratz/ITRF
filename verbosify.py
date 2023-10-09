import json
import openai

# Set your OpenAI API key
api_key = "YOUR_API_KEY"
openai.api_key = api_key

# Define a function to generate verbose responses using OpenAI's Chat API
def generate_verbose_response(question, context):
    # Build a prompt for the Chat API
    messages = [
        {"role":"system", "content": "You generate answers for the questions based on the given context. When the question cannot be answered please indicate by 'I do not know'"}
        {"role":"user", "content": f"Q: {question}\nC: {context}\nA:"}
    ]

    # Generate a response from the Chat API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150,  # You can adjust this for desired verbosity
        temperature=0.7,  # You can adjust this for creativity
        top_p=1.0,
        frequency_penalty=0.0,
        stop=None  # Let OpenAI decide when to stop
    )

    # Extract and return the generated answer
    answer = response.choices[0].text.strip()
    return answer

# Load the Squad dataset
with open('squad_dataset.json', 'r') as file:
    squad_data = json.load(file)

# Iterate through each Squad entry and generate verbose responses
verbose_squad_data = []
for entry in squad_data['data']:
    for paragraph in entry['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            verbose_answer = generate_verbose_response(question, context)
            qa['verbose_answer'] = verbose_answer
    verbose_squad_data.append(entry)

# Save the transformed dataset
with open('verbose_squad_dataset.json', 'w') as file:
    json.dump({'data': verbose_squad_data}, file)