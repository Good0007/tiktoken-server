'''
Rest service that exposes a single POST endpoint that accepts a JSON object 
with a  key, "prompt" and an optional key "model", and returns a JSON object with 
a single key, "tokens" # that has an array of the tokens returned by tiktoken.

Prompt is a string that needs to be tokenized. model is a string that is the name 
of a model in the  list of models returned by tiktoken.models(). If model is not
provided, use the default model "text-davinci-003". If the model is not found,
return an empty array. If the model is found, 
return the tokens returned by tiktoken.tokenize() for the prompt and model.

@Author: howdymic

'''

import json

from flask import Flask, request
import tiktoken

app = Flask(__name__)


@app.route("/tokenize/count", methods=['POST'])
def countTokenFromMsg():
    data = request.get_json()
    model = "gpt-3.5-turbo-0301"
    if 'model' in data:
        model = data['model']
    if not 'messages' in data:
        return json.dumps({'code': '0', 'count': 0})
    print('/tokenize/count - Request data:', data)
    count = 0
    try:
        count = num_tokens_from_messages(data['messages'], model)
    except:
        print("/tokenize/count - error: ", data)
        return json.dumps({'code': '0', 'error': 'parse error!', 'count': count})
    return json.dumps({'code': '0', 'count': count})


@app.route("/tokenize", methods=['POST', 'GET'])
def token_count():
    data = request.get_json()
    if not 'prompt' in data:
        return json.dumps({'tokens': []})
    model = "gpt-3.5-turbo-0301"

    if 'model' in data:
        model = data['model']
    print('/tokenize - Request data:', data)
    try:
        print('/tokenize - Getting encoding for model' + model)
        enc = tiktoken.encoding_for_model(model)
    except:
        return json.dumps({'code': '1', 'error': "Model not found"})
    return json.dumps({'code': '0', 'tokens': enc.encode(data['prompt'])})


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9999)

# python -m flask --app test.py run
#  curl -d '{"prompt" : "hello world", "model" : "text-davinci-003"}' -H "Content-Type: application/json"  http://localhost:5000/tokenize
