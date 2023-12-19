import dotenv
import bs4
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from store import Store

dotenv.load_dotenv()

class RAG:
    def __init__(self):
        self.store = Store("data/chroma", retriever_mode="parent")
        self.retriever = self.store.get_retriever()

        self.prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. \

        Question: {question} 

        Context: {context} 

        Answer:"
        """)

        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )


    def invoke(self, str):
        return self.rag_chain.invoke(str)

if __name__ == "__main__":
    rag = RAG()
    print(rag.invoke("What is Task Decomposition?"))
