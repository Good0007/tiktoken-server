from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFium2Loader
import json

# loader = PyPDFLoader("/Users/kangkang/Downloads/远航介绍问答.pdf")
# pages = loader.load_and_split()
# print(pages)
text_splitter_600 = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=200,
    separators=["。", "？", "！", "；"]
)

text_splitter_800 = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["。", "？", "！", "；"]
)

text_splitter_1000 = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["。", "？", "！", "；"]
)

## 解析文档段落
def split_documents(chunk_size = 600,file_path = ""):
    loader = PyPDFium2Loader(file_path)
    documents = loader.load()
    if chunk_size == 600:
        splitted_documents = text_splitter_600.split_documents(documents)
    if chunk_size == 800:
        splitted_documents = text_splitter_800.split_documents(documents)
    if chunk_size == 1000:
        splitted_documents = text_splitter_1000.split_documents(documents)
    # self.db = Chroma.from_documents(splitted_documents, self.embeddings).as_retriever()
    # self.chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    # print(splitted_documents)
    docs = []
    for doc in splitted_documents:
        content = {"content": doc.page_content, "page": doc.metadata['page']}
        docs.append(content)
    return docs;

#print(json.dumps(split_documents(file_path="/Users/kangkang/Downloads/远航介绍问答.pdf"), ensure_ascii=False))
