import json

from flask import Flask, request
import tiktoken_util
import text_splitter_util

app = Flask(__name__)
#app.run()

@app.route("/text/splitter", methods=['POST'])
def split_documents():
    data = request.get_json()
    if not 'filePath' in data:
        return json.dumps({'code': '1', 'error': "请传入文件路径或url地址！"})
    print('/text/splitter - Request data:', data)
    chunk_size = 600
    if 'chunkSize' in data:
        chunk_size = data['chunkSize']
    try:
        doc = text_splitter_util.split_documents(chunk_size,file_path=data['filePath'])
        return write_result({'code': '0', 'data': doc})
    except:
        print("/text/splitter - error: ", data)
        return write_result({'code': '1', 'error': "文件解析异常,请检查地址！"})


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
        count = tiktoken_util.num_tokens_from_messages(data['messages'], model)
        return write_result({'code': '0', 'count': count})
    except:
        print("/tokenize/count - error: ", data)
        return write_result({'code': '1', 'error': 'Model not found!', 'count': count})


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
        enc = tiktoken_util.tiktoken.encoding_for_model(model)
        return write_result({'code': '0', 'tokens': enc.encode(data['prompt'])})
    except:
        return json.dumps({'code': '1', 'error': "Model not found"})


# python -m flask --app test.py run
#  curl -d '{"prompt" : "hello world", "model" : "text-davinci-003"}' -H "Content-Type: application/json"  http://localhost:5000/tokenize

def write_result(message):
    return json.dumps(message,ensure_ascii=False)