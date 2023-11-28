from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseOutputParser
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

text_loader = TextLoader(os.path.join('facts.txt'))

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

text_chunk_documents = text_loader.load_and_split(
    text_splitter=text_splitter
)

embedding = OpenAIEmbeddings()

embedding_db = Chroma.from_documents(
    text_chunk_documents,
    embedding=embedding,
    persist_directory="emb_db"
)


chat_llm = ChatOpenAI()

chat_prompt = ChatPromptTemplate(
    input_variables=['fact', 'question'],
    messages=[
        SystemMessagePromptTemplate.from_template(
            "Use the following interesting fact to answer the users question : {fact}"
        ),
        HumanMessagePromptTemplate.from_template(
            "Here is the user's question: \n {question}"
        )
    ]
)

chat_chain = LLMChain(
    llm=chat_llm,
    prompt=chat_prompt
)

results = embedding_db.similarity_search(
    "What is an interseting fact about the english language?",
)

print(results)

for result in results:
    print(result.page_content)
    print('\n')

# output = chat_chain({
#     "fact": result[0].page_content,
#     "question": "What is an interseting fact about the english language?"
# })


