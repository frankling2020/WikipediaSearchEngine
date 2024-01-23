from collections import Counter, defaultdict
from indexing import InvertedIndex

"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
NOTE:
TODO's for hw2 are marked with `hw2` in the comments. See README.md for more details.
"""

import numpy as np
import math

class Ranker:
    '''
    The ranker class is responsible for generating a list of documents for a given query, ordered by their
    scores using a particular relevance function (e.g., BM25). A Ranker can be configured with any RelevanceScorer.
    '''

    # NOTE: (hw2) Note that `stopwords: set[str]` is a new parameter that you will need to use in your code.
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str], scorer: 'RelevanceScorer') -> None:
        '''
        Initializes the state of the Ranker object 
        '''
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        nonsense_token = ""
        self.stopwords_filter = lambda x: nonsense_token if str.lower(x) in stopwords else x

    # NOTE: (hw2): `query(self, query: str) -> list[dict]` is a new function that you will need to implement.
    #            see more in README.md.
    def query(self, query: str) -> list[dict]:
        '''
        Searches the collection for relevant documents to the query and returns a list 
        of documents ordered by their relevance (most relevant first).

        Args:
            query (str): The query to search for

        Returns:
            list: a list of dictionary objects with keys "docid" and "score" where docid is a
                  particular document in the collection and score is that document's relevance
        '''
        # TODO (hw1): Tokenize the query and remove stopwords, if needed
        results = []
        query_parts = self.tokenize(query)
        query_parts = list(map(self.stopwords_filter, query_parts))
     
        # TODO (hw2): Fetch a list of possible documents from the index and create a mapping from
        # a document ID to a dictionary of the counts of the query terms in that document.
        # You will pass the dictionary to the RelevanceScorer as input.
        #
        # NOTE: The accumulate_doc_term_counts() method for L2RRanker in l2r.py does something very
        # similar to what is needed for this step
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
        # TODO (hw1): Rank the documents using a RelevanceScorer (like BM25 from the below classes) 
        for docid in relevant_docs:
            doc_score = self.scorer.score(docid, doc_term_counts[docid], query_parts)
            results.append({'docid': docid, 'score': doc_score})
        results.sort(key=lambda x: (-x['score'], x['docid']))
        # TODO (hw1): Return the **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]
        return results


class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    # TODO(hw1): Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF)
    #      and not in this one

    def __init__(self, index: InvertedIndex, parameters) -> None:
        raise NotImplementedError

    # TODO (hw2): Note the change here: `score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float`
    #             See more in README.md.
    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        '''
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid (int): The ID of the document in the collection
            doc_word_counts: A dictionary containing all words in the document and their frequencies
            query_parts: A list of all the words in the query

        Returns:
            float: a score for how relevant the document is. Higher scores are more relevant.
        '''
        raise NotImplementedError


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        '''
        Scores all documents as 10
        '''
        # Print randomly ranked results
        return 10


# TODO (hw1): Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters = {}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        # 0. Get the word count vector of the document and the query
        query_parts_count = Counter(query_parts)
        tokens = set(query_parts_count.keys()).intersection(set(doc_word_counts.keys()))
        query_parts_count = {token: query_parts_count[token] for token in tokens}
        doc_word_counts = {token: doc_word_counts[token] for token in tokens}
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query
        if len(doc_word_counts) == 0:
            doc_score = 0
        else:
            doc_score = np.dot(list(doc_word_counts.values()), list(query_parts_count.values()))
        # 2. Return the score
        return doc_score


# TODO (hw1): Implement DirichletLM
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
        doc_score = sum(doc_score) if len(doc_score) != 0 else 0
        doc_score += len(query_parts) * math.log(mu / (document_length + mu))
        # 4. Return the score
        return doc_score


# TODO (hw1): Implement BM25
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
        doc_score = sum(doc_score) if len(doc_score) > 0 else 0
        # 4. Return the score
        return doc_score


# TODO (hw1): Implement Pivoted Normalization
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
        doc_score = sum(doc_score) if len(doc_score) > 0 else 0
        # 4. Return the score
        return doc_score


# TODO (hw1): Implement TF-IDF
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
        doc_score = sum(doc_score) if len(doc_score) > 0 else 0
        # 4. Return the score
        return doc_score


# TODO (hw1): Implement your own ranker with proper heuristics
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
        doc_score = sum(doc_score) if len(doc_score) != 0 else 0
        # 4. Return the score
        return doc_score
