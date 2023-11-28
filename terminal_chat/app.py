from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

chat_llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model='gpt-3.5-turbo',
    temperature=0.7
)

# output = chat_llm([
#     SystemMessage(content="You are chatbot specialize in Machine Learning"),
#     HumanMessage(content="Which is better classifiaction algorithm?")
# ])

chat_prompt = ChatPromptTemplate.from_messages([
    ('system', "You are chatbot specialize in {content}. Help user or assits them on their query."),
    ('human', "Find me some good {category} movies to watch. In comma seperated list")
])

chat_chain = chat_prompt | chat_llm

# response = chat_chain.invoke({
#     "content": "movies",
#     "category": "crime-thriller"
# })

prompt = ChatPromptTemplate(
    input_variables=['content', 'category'],
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are chatbot specialize in {content}. Help user or assits them on their query."
        ),
        HumanMessagePromptTemplate.from_template(
            "Write me short {category} poem for my partner"
        )
    ]
)

chain = prompt | chat_llm 

res = chain.invoke({
    "content": "Poetry",
    "category": "Romantic"
})

print(res)