from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from custom_filter_retriever import CustomFilterRetriever
import langchain
from dotenv import load_dotenv
import os

langchain.debug = True
load_dotenv()

# text_loader = TextLoader(os.path.join('facts.txt'))

# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=200,
#     chunk_overlap=0
# )

# text_chunk_documents = text_loader.load_and_split(
#     text_splitter=text_splitter
# )

chat_llm = ChatOpenAI()

text_embeddings = OpenAIEmbeddings()

embedding_db = Chroma(
    persist_directory='emb_db',
    embedding_function=text_embeddings
)

text_emb_retriever = embedding_db.as_retriever()

cust_retriever = CustomFilterRetriever(
    embeddings=text_embeddings,
    chroma=embedding_db
)

chain = RetrievalQA.from_chain_type(
    llm=chat_llm,
    retriever=cust_retriever,
    chain_type='stuff'
)

result = chain.run("What is an interesting fact about the english language?")

print(result)


