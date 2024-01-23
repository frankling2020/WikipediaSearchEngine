"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import CrossEncoder
from indexing import InvertedIndex
import math
import torch

class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    # TODO: Implement this class properly; this is responsible for returning a list of sorted relevant documents
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str], scorer: 'RelevanceScorer') -> None:
        """
        Initializes the state of the Ranker object.

        TODO (HW3): Previous homeworks had you passing the class of the scorer to this function
        This has been changed as it created a lot of confusion.
        You should now pass an instantiated RelevanceScorer to this function.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        nonsense_token = ""
        self.stopwords_filter = lambda x: nonsense_token if str.lower(x) in stopwords else x

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A list of dictionary objects with keys "docid" and "score" where docid is
            a particular document in the collection and score is that document's relevance

        TODO (HW3): We are standardizing the query output of Ranker to match with L2RRanker.query and VectorRanker.query
        The query function should return a sorted list of tuples where each tuple has the first element as the document ID
        and the second element as the score of the document after the ranking process.
        """
        # TODO: Tokenize the query and remove stopwords, if needed
        results = []
        query_parts = self.tokenize(query)
        query_parts = list(map(self.stopwords_filter, query_parts))
        # TODO: Fetch a list of possible documents from the index and create a mapping from
        #       a document ID to a dictionary of the counts of the query terms in that document.
        #       You will pass the dictionary to the RelevanceScorer as input.
        relevant_docs = set()
        doc_term_counts = defaultdict(Counter)
        for token in set(query_parts):
            doc_ids = set()
            for doc_info in self.index.get_postings(token):
                doc_id = doc_info[0]
                term_freq = doc_info[1]
                doc_term_counts[doc_id][token] = term_freq
                doc_ids.add(doc_id)
            relevant_docs.update(set(doc_ids))
        # TODO: Rank the documents using a RelevanceScorer (like BM25 from below classes) 
        for docid in relevant_docs:
            doc_score = self.scorer.score(docid, doc_term_counts[docid], query_parts)
            results.append((docid, doc_score))
        results.sort(key=lambda x: (-x[1], x[0]))
        # TODO: Return the **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]
        return results


class RelevanceScorer:
    """
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    """
    # TODO (HW1): Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM,
    #             BM25, PivotedNormalization, TF_IDF) and not in this one
    def __init__(self, index: InvertedIndex, parameters) -> None:
        raise NotImplementedError

    # NOTE (hw2): Note the change here: `score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float`
    #             See more in README.md.
    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies
            query_parts: A list of all the words in the query
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)
        """
        raise NotImplementedError


# TODO (HW1): Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query
        query_parts_count = Counter(query_parts)
        tokens = set(query_parts_count.keys()).intersection(set(doc_word_counts.keys()))
        query_parts_count = {token: query_parts_count[token] for token in tokens}
        doc_word_counts = {token: doc_word_counts[token] for token in tokens}
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query
        score = 0
        if len(doc_word_counts) > 0:
            score = np.dot(list(doc_word_counts.values()), list(query_parts_count.values()))
        # 2. Return the score
        return score  # (`score` should be defined in your code; you can call it whatever you want)


# TODO (HW1): Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        # 1. Get necessary information from index
        query_parts_count = Counter(query_parts)
        tokens = set(query_parts_count.keys()).intersection(set(doc_word_counts.keys()))
        query_parts_count = {token: query_parts_count[token] for token in tokens}
        doc_word_counts = {token: doc_word_counts[token] for token in tokens}
        # 2. Compute additional terms to use in algorithm
        doc_statistics = self.index.get_statistics()
        collection_length = doc_statistics.get('total_token_count', 0)
        total_term_freqs = {}
        for token in doc_word_counts:
            term_info = self.index.get_term_metadata(token)
            total_term_freqs[token] = term_info['term_freq'] if term_info is not None else 0
        collection_term_count = {token: total_term_freqs.get(token, 0) for token in doc_word_counts}
        document_length = self.index.get_doc_metadata(docid).get('length', 0)
        mu = self.parameters['mu']
        # 3. For all query_parts, compute score
        doc_score = [
            query_parts_count[token] * math.log(1 + (doc_word_counts[token] * collection_length / mu / collection_term_count[token]))
                for token in doc_word_counts
        ]
        score = sum(doc_score) if len(doc_score) != 0 else 0
        score += len(query_parts) * math.log(mu / (document_length + mu))
        # 4. Return the score
        return score  # (`score` should be defined in your code; you can call it whatever you want)


# TODO (HW1): Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        # 0. Get the word count vector of the document and the query
        query_parts_count = Counter(query_parts)
        tokens = set(query_parts_count.keys()).intersection(set(doc_word_counts.keys()))
        query_parts_count = {token: query_parts_count[token] for token in tokens}
        doc_word_counts = {token: doc_word_counts[token] for token in tokens}
        # 1. Get necessary information from index
        document_length = self.index.get_doc_metadata(docid).get('length', 0)
        doc_statistics = self.index.get_statistics()
        document_average_length = doc_statistics.get('mean_document_length', 0)
        number_of_documents = doc_statistics.get('number_of_documents', 0)
        document_appearance = {}
        for token in doc_word_counts:
            term_info = self.index.get_term_metadata(token)
            if term_info is not None:
                document_appearance[token] = term_info['doc_freq']
        # 2. Compute additional terms to use in algorithm
        doc_tf_down = 1 - self.b + self.b * (document_length / document_average_length)
        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
        doc_tf = {token: (self.k1 + 1) * doc_word_counts[token] / (self.k1 * doc_tf_down + doc_word_counts[token]) for token in doc_word_counts}
        doc_idf = {token: math.log((number_of_documents - document_appearance[token] + 0.5) / (document_appearance[token] + 0.5)) for token in doc_word_counts}
        qtf = {token: ((self.k3 + 1) * query_parts_count[token]) / (self.k3 + query_parts_count[token]) for token in query_parts_count}
        doc_score = [doc_tf[token] * doc_idf[token] * qtf[token] for token in doc_word_counts]
        score = sum(doc_score) if len(doc_score) > 0 else 0
        # 4. Return the score
        return score  # (`score` should be defined in your code; you can call it whatever you want)


# TODO (HW1): Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        query_parts_count = Counter(query_parts)
        tokens = set(query_parts_count.keys()).intersection(set(doc_word_counts.keys()))
        query_parts_count = {token: query_parts_count[token] for token in tokens}
        doc_word_counts = {token: doc_word_counts[token] for token in tokens}
        # 1. Get necessary information from index
        document_length = self.index.get_doc_metadata(docid).get('length', 0)
        doc_statistics = self.index.get_statistics()
        document_average_length = doc_statistics.get('mean_document_length', 0)
        number_of_documents = doc_statistics.get('number_of_documents', 0)
        document_appearance = {}
        for token in doc_word_counts:
            term_info = self.index.get_term_metadata(token)
            if term_info is not None:
                document_appearance[token] = term_info['doc_freq']
        # 2. Compute additional terms to use in algorithm
        doc_tf_down = 1 - self.b + self.b * (document_length / document_average_length)
        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
        doc_tf = {token: (1 + math.log(1 + math.log(doc_word_counts[token]))) / doc_tf_down for token in doc_word_counts}
        doc_idf = {token: math.log((number_of_documents + 1) / document_appearance[token]) for token in doc_word_counts}
        doc_score = [query_parts_count[token] * doc_tf[token] * doc_idf[token] for token in doc_word_counts]
        score = sum(doc_score) if len(doc_score) > 0 else 0
        # 4. Return the score
        return score  # (`score` should be defined in your code; you can call it whatever you want)


# TODO (HW1): Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        # 1. Get necessary information from index
        query_parts_count = Counter(query_parts)
        tokens = set(query_parts_count.keys()).intersection(set(doc_word_counts.keys()))
        query_parts_count = {token: query_parts_count[token] for token in tokens}
        doc_word_counts = {token: doc_word_counts[token] for token in tokens}
        # 2. Compute additional terms to use in algorithm
        number_of_documents = self.index.get_statistics().get('number_of_documents', 1)
        document_appearance = {}
        for token in doc_word_counts:
            term_info = self.index.get_term_metadata(token)
            if term_info is not None:
                document_appearance[token] = term_info['doc_freq']
        # 3. For all query parts, compute the TF and IDF to get a score
        doc_tf = {token: math.log(doc_word_counts[token] + 1) for token in doc_word_counts}
        doc_idf = {token: 1 + math.log(number_of_documents / document_appearance[token]) for token in doc_word_counts}
        doc_score = [doc_tf[token] * doc_idf[token] for token in doc_word_counts]
        score = sum(doc_score) if len(doc_score) > 0 else 0
        # 4. Return the score
        return score  # (`score` should be defined in your code; you can call it whatever you want)


# TODO (HW3): The CrossEncoderScorer class uses a pre-trained cross-encoder model from the Sentence Transformers package
#             to score a given query-document pair; check README for details
#
# NOTE: This is not a RelevanceScorer object because the method signature for score() does not match, but it
# has the same intent, in practice
class CrossEncoderScorer:
    '''
    A scoring object that uses cross-encoder to compute the relevance of a document for a query.
    '''
    def __init__(self, raw_text_dict: dict[int, str], 
                 cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model
        """
        # TODO: Save any new arguments that are needed as fields of this class
        self.cross_encoder = CrossEncoder(cross_encoder_model_name, max_length=512)
        self.raw_text_dict = raw_text_dict

    @torch.no_grad()
    def score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.
        
        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        # NOTE: Do not forget to handle an edge case
        # (e.g., docid does not exist in raw_text_dict or empty query, both should lead to 0 score)
        score = 0
        if len(query) != 0 and docid in self.raw_text_dict:
        # NOTE: unlike the other scorers like BM25, this method takes in the query string itself,
        # not the tokens!
            score = self.cross_encoder.predict([[query, self.raw_text_dict[docid]]])[0]
        # TODO (HW3): Get a score from the cross-encoder model
        #             Refer to IR_Encoder_Examples.ipynb in Demos folder on Canvas if needed
        return score


# TODO (HW1): Implement your own ranker with proper heuristics
class YourRanker(RelevanceScorer):
    def __init__(self, index, parameters: dict = {'lambda': 0.9, 'alpha': 0.5}) -> None:
        self.index = index
        self.parameters = parameters
    
    def score(self, docid: int,  doc_word_counts: dict[str, int], query_parts: list[str]) -> dict[str, int]:
        # 1. Get necessary information from index
        query_parts_count = Counter(query_parts)
        tokens = set(query_parts_count.keys()).intersection(set(doc_word_counts.keys()))
        query_parts_count = {token: query_parts_count[token] for token in tokens}
        doc_word_counts = {token: doc_word_counts[token] for token in tokens}
        # 2. Compute additional terms to use in algorithm
        doc_statistics = self.index.get_statistics()
        collection_length = doc_statistics.get('total_token_count', 0)
        total_term_freqs = {}
        for token in doc_word_counts:
            term_info = self.index.get_term_metadata(token)
            total_term_freqs[token] = term_info['term_freq'] if term_info is not None else 0
        document_length = self.index.get_doc_metadata(docid).get('length', 0)
        collection_term_freqs = {token: document_length * total_term_freqs[token] / collection_length 
                for token in doc_word_counts}
        alpha = self.parameters['alpha']
        mixed_probs = {token: (1 + alpha) * (doc_word_counts[token]/collection_term_freqs[token]) - alpha for token in doc_word_counts}
        mu = 1 / self.parameters['lambda'] - 1

        # 3. For all query_parts, compute score
        doc_score = [
            query_parts_count[token] * math.log(1 + mu * max(mixed_probs[token], 0))
                for token in doc_word_counts
        ]
        score = sum(doc_score) if len(doc_score) != 0 else 0
        # 4. Return the score
        return score


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10
        """
        # Print randomly ranked results
        return 10

