import tiktoken

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
    elif model == "text-davinci-002" or model == "text-davinci-003":
        encoding = tiktoken.get_encoding("p50k_base")
        tokens_per_message = 4
        tokens_per_name = -1
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

text = '''
一个大语言模型的技术分享，请帮我介绍下大语言模型的基本原理、训练方法、训练数据要求、微调方法等内容。->好的，请听我介绍：
### 基本原理
大语言模型的基本原理是使用深度学习算法学习一个能够预测下一个单词的模型。具体来说，大语言模型将单词序列作为输入，输出下一个可能出现的单词。这种模型可以被用于自然语言处理中的各种任务，如语言生成、文本分类、文本相似度匹配等。
### 训练方法
大语言模型的训练方法一般是基于神经网络的语言模型，即使用循环神经网络（RNN）或者变种（如LSTM或GRU）建立模型，通过最小化损失函数来优化模型参数。训练过程中还可以使用一些技巧，如teacher forcing、dropout等，以提高模型的泛化能力和训练效果。
### 训练数据要求
大语言模型的训练需要大量的文本数据，数据要求具有足够的多样性和覆盖面，可以来自于社交媒体、新闻、维基百科等多个来源。同时，在训练数据的预处理过程中，可以进行分词、词性标注、去除停用词等处理，以提高模型的训练效果。
### 微调方法
在实际应用中，大语言模型通常需要进行微调才能适应特定的应用场景。微调方法通常是将已经训练好的模型在新的数据集上进行训练，以调整模型参数并提高模型的性能。同时，还可以对微调后的模型进行 fine-tuning，即在已有模型基础上进一步训练以获得更好的效果。
以上是大语言模型的基本原理、训练方法、训练数据要求和微调方法，希望对您有所帮助。
'''

# print(num_tokens_from_messages([{"role":"user","message":text}],"gpt-4"))
