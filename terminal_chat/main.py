from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()

chat_llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model='gpt-3.5-turbo',
    temperature=0.7
)

memory = ConversationBufferMemory(
    memory_key="messages",
    chat_memory= FileChatMessageHistory('messages.json'),
    return_messages=True
)

chat_prompt = ChatPromptTemplate(
    input_variables=['content'],
    messages=[
        MessagesPlaceholder(variable_name='messages'),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chat_chain = LLMChain(
    llm=chat_llm,
    prompt=chat_prompt,
    output_key='message',
    memory=memory
)



if __name__ == '__main__':
    while True:
        content = input(" >> ")

        response = chat_chain({
            "content": content
        })

        print(response["message"])