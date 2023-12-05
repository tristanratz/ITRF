
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory

class RAG:

    def __init__(self, system=None, ):

        # LLM
        self.llm = ChatOpenAI(temperature=0)
 # k=10
        
        template = """Answer the question based only on the following context:

        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a assistant. Answer the questions based only on the provided context."
                ),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("""
                                                         Context: {context}
                                                         Question: {question}
                                                         """)
            ]
        )

        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def infernce(self, prompt):
        return self.chain.invoke(prompt)