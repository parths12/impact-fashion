import faiss
import numpy as np
import pandas as pd

class FaissRecommender:
    def __init__(self, catalog_csv):
        self.catalog = pd.read_csv(catalog_csv)
        self.embeddings = np.vstack(self.catalog['embedding'].apply(lambda x: np.fromstring(x, sep=',')))
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query_embedding, top_k=5):
        D, I = self.index.search(query_embedding.reshape(1, -1), top_k)
        return self.catalog.iloc[I[0]].assign(distance=D[0]) 