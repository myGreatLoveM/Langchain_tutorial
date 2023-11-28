from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv
import os

load_dotenv()

chat_llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model='gpt-3.5-turbo',
    temperature=0.7,
    verbose=True
)

llm = OpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model='text-davinci-003',
    temperature=0.7,
    verbose=True
)

memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=llm
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
    memory=memory,
    verbose=True
)



if __name__ == '__main__':
    while True:
        content = input(" >> ")

        response = chat_chain({
            "content": content
        })

        print(response["message"])