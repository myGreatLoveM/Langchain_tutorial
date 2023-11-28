from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever

class CustomFilterRetriever(BaseRetriever):
    embeddings : Embeddings
    chroma : Chroma

    def get_relevant_documents(self, query):
        query_emb = self.embeddings.embed_query(query)

        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=query_emb,
            lambda_mult=0.8
        )
    
    async def aget_relevant_documents(self):
        return []