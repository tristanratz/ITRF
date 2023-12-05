from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

embeddings = OpenAIEmbeddings()
raw_documents = TextLoader('./data/test.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, embeddings)

db.save_local("./data/faiss_index")

db = FAISS.load_local("./data/faiss_index", embeddings)