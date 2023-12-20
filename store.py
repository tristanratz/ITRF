import os
import bs4
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.storage import LocalFileStore
from langchain.storage import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from embeddings.dragon import DRAGON

load_dotenv()

class Store:
    db = None
    path: str = ""
    retriever = None
    store = None

    def __init__(self, path, docs=None, embedding="OpenAI", retriever_mode="chunk", chunk_size=2000, chunk_overlap=200):
        self.path = path


        if embedding == "OpenAI":
            self.embedding = OpenAIEmbeddings()
        elif embedding == "bge":
            # https://python.langchain.com/docs/integrations/text_embedding/bge_huggingface
            self.embedding = HuggingFaceBgeEmbeddings(model_name="./models/retriever/bge-base-en-v1.5")
        elif embedding == "dragon":
            self.embedding = DRAGON()


        if os.path.isdir(os.path.join(path, "./")):
            self.db = Chroma(
                        embedding_function=self.embedding,
                        persist_directory=self.path
                    )
            if retriever_mode == "parent":
                self.store = InMemoryStore() #LocalFileStore(path)
                child_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=chunk_overlap)
                parent_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                self.retriever = ParentDocumentRetriever(
                    vectorstore=self.db,
                    docstore=self.store,
                    child_splitter=child_splitter,
                    parent_splitter=parent_splitter,
                )
            else:
                self.retriever = self.db.as_retriever()
        else:
            self.init_database(retriever_mode=retriever_mode)

    """
    retriever_mode: 
        chunk - Only the encoded chunk
        parent - A substentially larger chunk
    """
    def init_database(self, documents=None, retriever_mode="chunk", chunk_size=2000, chunk_overlap=200):
        print("Init database...")
        
        docs = documents

        if docs is None:
            loader = WebBaseLoader(
                web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    )
                ),
            )
            docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        if retriever_mode == "chunk":
            splits = text_splitter.split_documents(docs)

            # load dataset
            self.db = Chroma.from_documents(splits, self.embedding, persist_directory=self.path)
            self.retriever = self.db.as_retriever()
        elif retriever_mode == "parent":
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

            self.db = Chroma(
                embedding_function=self.embedding,
                persist_directory=self.path
            )

            # self.store = LocalFileStore(self.path)
            self.store = InMemoryStore()
            self.retriever = ParentDocumentRetriever(
                vectorstore=self.db,
                docstore=self.store,
                child_splitter=child_splitter,
                parent_splitter=text_splitter,
            )
            self.retriever.add_documents(docs, ids=None)
        self.save()
        print("Initialization completed.")

    def save(self):
        # self.db.persist()
        return

    def get_retriever(self):
        return self.retriever

    def add_docs(self, docs):
        #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # documents = text_splitter.split_documents(raw_documents)
        self.db.add_document()
        self.save()

    def search(self, query):
        vec = self.embed_query(query)
        return self.search_vec(vec)

    def search_vec(self, vec):
        return self.db.similarity_search_by_vector(vec)

    def embed_docs(self, docs):
        return self.embedding.embed(docs)

    def embed_query(self, query):
        return self.embedding.embed_query(query)

if __name__ == "__main__":
    store = Store("data/chroma", retriever_mode="parent")
    print(store.get_retriever().invoke("What is Task Decomposition?"))