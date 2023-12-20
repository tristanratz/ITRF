import dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from store import Store
from model import Model

dotenv.load_dotenv()

class RAG:
    def __init__(self, embedding="OpenAI"):

        self.store = Store("data/chroma", retriever_mode="parent", embedding="bge")
        self.retriever = self.store.get_retriever()

        self.model = Model(model="llama")
        self.prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. \

        Question: {question} 

        Context: {context} 

        Answer:"
        """)

        self.llm = self.model.get_llm()

        self.llm_chain = (
            self.llm
            | StrOutputParser() 
            | RunnablePassthrough()
        )

    def _format_docs(self, docs):
            return "\n\n".join(doc.page_content for doc in docs)

    def invoke(self, str):
        docs = self.retriever.invoke(str)
        print(docs)
        prompt = self.prompt.invoke(
            {"context": self._format_docs(docs), "question": str}
            )
        
        return self.llm_chain.invoke(prompt)

if __name__ == "__main__":
    rag = RAG()
    print(rag.invoke("What is Task Decomposition?"))
