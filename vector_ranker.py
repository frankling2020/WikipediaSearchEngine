from sentence_transformers import SentenceTransformer, util
from numpy import ndarray
from ranker import Ranker
import torch
import numpy as np


class VectorRanker(Ranker):
    def __init__(self, bi_encoder_model_name: str, encoded_docs: ndarray,
                 row_to_docid: list[int]) -> None:
        """
        Initializes a VectorRanker object.

        Args:
            bi_encoder_model_name: The name of a huggingface model to use for initializing a 'SentenceTransformer'
            encoded_docs: A matrix where each row is an already-encoded document, encoded using the same encoded
                as specified with bi_encoded_model_name
            row_to_docid: A list that is a mapping from the row number to the document id that row corresponds to
                the embedding

        Using zip(encoded_docs, row_to_docid) should give you the mapping between the docid and the embedding.
        """
        # Use device='cpu' when doing model instantiation (for AG)
        # If you know what the parameter does, feel free to play around with it
        # TODO: Instantiate the bi-encoder model here
        self.device = torch.device('cpu')  # Do not change this unless you know what you are doing
        self.bi_encoder = SentenceTransformer(bi_encoder_model_name, device=self.device)
        # self.doc_to_encoded_doc = dict(zip(row_to_docid, encoded_docs))
        self.encoded_docs = encoded_docs
        self.row_to_docid = row_to_docid

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Encodes the query and then scores the relevance of the query with all the documents.

        Args:
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A sorted list of tuples containing the document id and its relevance to the query,
            with most relevant documents first
        """
        # NOTE: Do not forget to handle edge cases
        doc_id_score = []
        if len(query) != 0 and len(self.row_to_docid) != 0:
            # TODO: Encode the query using the bi-encoder
            encoded_query = self.bi_encoder.encode(query, device=self.device, normalize_embeddings=True)
            # TODO: Score the similarity of the query vector and document vectors for relevance
            # Calculate the dot products between the query embedding and all document embeddings
            similarity_scores = np.sum(encoded_query * self.encoded_docs, axis=1)
            # TODO: Generate the ordered list of (document id, score) tuples
            doc_id_score = list(zip(self.row_to_docid, similarity_scores))
            # TODO: Sort the list so most relevant are first
            doc_id_score.sort(key=lambda x: (-x[1], x[0]))
            
        return doc_id_score
