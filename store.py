import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
#from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Refine https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever

class Store:
    db = None
    path: str = ""
    embeddings_model = OpenAIEmbeddings()
    query_embeddings_model = OpenAIEmbeddings()

    def __init__(self, path, library="FAISS"):
        self.path = path
        if os.path.isfile(os.path.join(path, "index.faiss")):
            self.db = FAISS.load_local(path, self.embeddings)
        else:
            self.init_database()

    def init_database(self):
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # load dataset
        self.db = FAISS.from_documents(splits, self.embeddings_model)
        self.retriever = FAISS.as_retriever()

    def add_docs(self, docs):
        #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # documents = text_splitter.split_documents(raw_documents)
        self.db.add_document()
        self.db.save_local(self.path)

    def search(self, query):
        vec = self.embed_query(query)
        return self.search_vec(vec)

    def search_vec(self, vec):
        return self.db.similarity_search_by_vector(vec)

    def embed_docs(self, docs):
        return self.embeddings_model.embed(docs)

    def embed_query(self, query):
        return self.query_embeddings_model.embed_query(query)

if __name__ == "__main__":
    store = Store("data/faiss_index")