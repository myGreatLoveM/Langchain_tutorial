from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

def text_to_chunks(filepath):
    text_loader = TextLoader(filepath)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=0
    )

    docs = text_loader.load_and_split(
        text_splitter=text_splitter
    )

    # for doc in docs:
    #     print(doc.page_content)
    #     print("\n")

    return docs

def generate_embeddings(docs, db_name):
    embeddings = OpenAIEmbeddings()

    if(os.path.exists(db_name)):
        db = Chroma(
            persist_directory=db_name,
            embedding_function=embeddings
        )
        return db
    else:
        db = Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory=db_name
        )
        return db


if __name__ == "__main__":
    chat_llm = ChatOpenAI(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0.7
    )

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

    documents = text_to_chunks('facts.txt')

    db = generate_embeddings(documents, 'embs')

    result = db.similarity_search(
        "What is an interseting fact about the english language?",
        k=1
    )

    output = chat_chain({
        "fact": result[0].page_content,
        "question": "What is an interseting fact about the english language?"
    })

    print(output['text'])
    
