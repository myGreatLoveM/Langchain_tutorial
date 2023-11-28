from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = OpenAIEmbeddings()

loader = TextLoader(os.path.join('facts.txt'))

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

docs = loader.load_and_split(
    text_splitter=text_splitter
)

for doc in docs:
    print(doc.page_content)
    print("\n")



db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="embs"
)

results = db.similarity_search_with_score("What is an interesting fact about the english language?")

for result in results:
    print(result[1])
    print(result[0].page_content)
    print("\n")