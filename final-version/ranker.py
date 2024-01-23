"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
from sample_data import SAMPLE_DOCS  # sample document import
from collections import Counter
import math
import numpy as np

class Ranker:
    # TODO implement this class properly. This is responsible for returning a list of sorted relevant documents.
    def __init__(self, index, document_preprocessor, stopword_filtering: bool, scorer: 'RelevanceScorer') -> None:
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer(self.index)
        self.stopword_filtering = stopword_filtering
        self.stopwords_filter = lambda x: x
        if self.stopword_filtering:
            with open('stopwords.txt', 'r') as stopwords_file:
                stopwords = list(map(lambda x: x.strip(), stopwords_file.readlines()))
            stopwords = set(stopwords)
            nonsense_token = ""
            self.stopwords_filter = lambda x: nonsense_token if str.lower(x) in stopwords else x

    def query(self, query: str) -> list[dict[str, int]]:        
        # 1. Tokenize query
        # 2. Fetch a list of possible documents from the index
        # 2. Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes)
        # 3. Return **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]
        results = []
        query_parts = self.tokenize(query)
        if self.stopword_filtering:
            query_parts = list(map(self.stopwords_filter, query_parts))
        relevant_docs = set()
        for token in query_parts:
            doc_ids = [doc_info[0] for doc_info in self.index.get_postings(token)]
            relevant_docs.update(set(doc_ids))
        for docid in relevant_docs:
            doc_score = self.scorer.score(docid=docid, query_parts=query_parts)
            results.append(doc_score)
        results.sort(key=lambda x: (-x['score'], x['docid']))
        return results


class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    # TODO Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) 
    #      and not in this one
    
    def __init__(self, index, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, query_parts: list[str]) -> dict[str, int]:
        raise NotImplementedError
    
    def get_doc_term_freq(self, docid: int, term: str) -> int:
        if term in self.index.index:
            pos = self.index.search_doc_pos(docid, term)
            if pos != -1:
                return self.index.index[term][pos][1]
        return 0


class SampleScorer(RelevanceScorer):
    def __init__(self, index, parameters) -> None:
        pass

    def score(self, docid: int, query_parts: list[str]) -> dict[str, int]:

        # Print randomly ranked results
        return {'docid': docid, 'score': 10}


# TODO Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
    
    def score(self, docid: int, query_parts: list[str]) -> dict[str, int]:
        # 0. Get the word count vector of the document and the query
        query_parts_count = Counter(query_parts)
        doc_word_count = {token: self.get_doc_term_freq(docid, token) for token in query_parts_count}
        # filter out words that don't appear in the document
        query_parts_count = {token: query_parts_count[token] for token in doc_word_count}
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query
        if len(doc_word_count) == 0:
            doc_score = 0
        else:
            doc_score = np.dot(list(doc_word_count.values()), list(query_parts_count.values()))
        # 2. Return the score
        return {'docid': docid, 'score': doc_score}


# TODO: Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index, parameters: dict = {'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, query_parts: list[str]) -> dict[str, int]:
        # 1. Get necessary information from index
        query_parts_count = Counter(query_parts)
        doc_word_count = {token: self.get_doc_term_freq(docid, token) for token in query_parts_count}
        # filter out words that don't appear in the document
        doc_word_count = {token: count for token, count in doc_word_count.items() if count != 0}
        query_parts_count = {token: query_parts_count[token] for token in doc_word_count}
        # 2. Compute additional terms to use in algorithm
        doc_statistics = self.index.get_statistics()
        collection_length = doc_statistics.get('total_token_count', 0)
        total_term_freqs = doc_statistics.get('vocab', {})
        collection_term_count = {token: total_term_freqs.get(token, 0) for token in doc_word_count}
        document_length = self.index.get_doc_metadata(docid).get('length', 0)
        mu = self.parameters['mu']
        # 3. For all query_parts, compute score
        # no idea mu can be 0
        doc_score = [
            query_parts_count[token] * math.log(1 + (doc_word_count[token] * collection_length / mu / collection_term_count[token]))
                for token in doc_word_count
        ]
        doc_score = sum(doc_score) if len(doc_score) != 0 else 0
        doc_score += len(query_parts) * math.log(mu / (document_length + mu))
        # 4. Return the score
        return {'docid': docid, 'score': doc_score}
        

# TODO: Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.parameters = parameters
    
    def score(self, docid: int, query_parts: list[str]) -> dict[str, int]:
        # Get parameters
        b = self.parameters['b']
        k1 = self.parameters['k1']
        k3 = self.parameters['k3']
        # 0. Get the word count vector of the document and the query
        query_parts_count = Counter(query_parts)
        doc_word_count = {token: self.get_doc_term_freq(docid, token) for token in query_parts_count}
        # filter out words that don't appear in the document
        doc_word_count = {token: count for token, count in doc_word_count.items() if count != 0}
        query_parts_count = {token: query_parts_count[token] for token in doc_word_count}
        # 1. Get necessary information from index
        document_length = self.index.get_doc_metadata(docid).get('length', 0)
        doc_statistics = self.index.get_statistics()
        document_average_length = doc_statistics.get('mean_document_length', 0)
        number_of_documents = doc_statistics.get('number_of_documents', 0)
        document_appearance = {}
        for token in doc_word_count:
            term_info = self.index.get_term_metadata(token)
            if term_info is not None:
                document_appearance[token] = term_info['doc_freq']
        # 2. Compute additional terms to use in algorithm
        doc_tf_down = 1 - b + b * (document_length / document_average_length)
        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
        doc_tf = {token: (k1 + 1) * doc_word_count[token] / (k1 * doc_tf_down + doc_word_count[token]) for token in doc_word_count}
        doc_idf = {token: math.log((number_of_documents - document_appearance[token] + 0.5) / (document_appearance[token] + 0.5)) for token in doc_word_count}
        qtf = {token: ((k3 + 1) * query_parts_count[token]) / (k3 + query_parts_count[token]) for token in query_parts_count}
        doc_score = [doc_tf[token] * doc_idf[token] * qtf[token] for token in doc_word_count]
        doc_score = sum(doc_score) if len(doc_score) > 0 else 0
        # 4. Return the score
        return {'docid': docid, 'score': doc_score}


# TODO: Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index, parameters: dict = {'b': 0.2}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, query_parts: list[str]) -> dict[str, int]:
        query_parts_count = Counter(query_parts)
        doc_word_count = {token: self.get_doc_term_freq(docid, token) for token in query_parts_count}
        # filter out words that don't appear in the document
        doc_word_count = {token: count for token, count in doc_word_count.items() if count != 0}
        query_parts_count = {token: query_parts_count[token] for token in doc_word_count}
        # 1. Get necessary information from index
        document_length = self.index.get_doc_metadata(docid).get('length', 0)
        doc_statistics = self.index.get_statistics()
        document_average_length = doc_statistics.get('mean_document_length', 0)
        number_of_documents = doc_statistics.get('number_of_documents', 0)
        document_appearance = {}
        for token in doc_word_count:
            term_info = self.index.get_term_metadata(token)
            if term_info is not None:
                document_appearance[token] = term_info['doc_freq']
        # 2. Compute additional terms to use in algorithm
        b = self.parameters['b']
        doc_tf_down = 1 - b + b * (document_length / document_average_length)
        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
        doc_tf = {token: (1 + math.log(1 + math.log(doc_word_count[token]))) / doc_tf_down for token in doc_word_count}
        doc_idf = {token: math.log((number_of_documents + 1) / document_appearance[token]) for token in doc_word_count}
        doc_score = [query_parts_count[token] * doc_tf[token] * doc_idf[token] for token in doc_word_count]
        doc_score = sum(doc_score) if len(doc_score) > 0 else 0
        # 4. Return the score
        return {'docid': docid, 'score': doc_score}


# TODO: Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
    
    def score(self, docid: int, query_parts: list[str]) -> dict[str, int]:
        # 1. Get necessary information from index
        query_parts_count = Counter(query_parts)
        doc_word_count = {token: self.get_doc_term_freq(docid, token) for token in query_parts_count}
        # filter out words that don't appear in the document
        doc_word_count = {token: count for token, count in doc_word_count.items() if count != 0}
        query_parts_count = {token: query_parts_count[token] for token in doc_word_count}
        # 2. Compute additional terms to use in algorithm
        number_of_documents = self.index.get_statistics().get('number_of_documents', 1)
        document_appearance = {}
        for token in doc_word_count:
            term_info = self.index.get_term_metadata(token)
            if term_info is not None:
                document_appearance[token] = term_info['doc_freq']
            else:
                print(token, doc_word_count[token], query_parts_count[token])
                print(self.index.index.get(token, {}))
        # 3. For all query parts, compute the TF and IDF to get a score
        doc_tf = {token: math.log(doc_word_count[token] + 1) for token in doc_word_count}
        # query_tf = {token: math.log(query_parts_count[token] + 1) for token in query_parts_count}
        doc_idf = {token: 1 + math.log(number_of_documents / document_appearance[token]) for token in doc_word_count}
        # doc_score = [doc_tf[token] * query_tf[token] * doc_idf[token] for token in doc_word_count]
        doc_score = [doc_tf[token] * doc_idf[token] for token in doc_word_count]
        doc_score = sum(doc_score) if len(doc_score) > 0 else 0
        # 4. Return the score
        return {'docid': docid, 'score': doc_score}


# TODO: Implement your own ranker with proper heuristics
class YourRanker(RelevanceScorer):
    def __init__(self, index, parameters: dict = {'lambda': 0.9, 'alpha': 0.1}) -> None:
        self.index = index
        self.parameters = parameters
    
    def score(self, docid: int, query_parts: list[str]) -> dict[str, int]:
        # 1. Get necessary information from index
        query_parts_count = Counter(query_parts)
        doc_word_count = {token: self.get_doc_term_freq(docid, token) for token in query_parts_count}
        # filter out words that don't appear in the document
        doc_word_count = {token: count for token, count in doc_word_count.items() if count != 0}
        query_parts_count = {token: query_parts_count[token] for token in doc_word_count}
        # 2. Compute additional terms to use in algorithm
        doc_statistics = self.index.get_statistics()
        collection_length = doc_statistics.get('total_token_count', 0)
        total_term_freqs = doc_statistics.get('vocab', {})
        collection_term_probs = {token: total_term_freqs[token] / collection_length for token in doc_word_count}
        document_length = self.index.get_doc_metadata(docid).get('length', 0)
        document_term_prob = {token: doc_word_count[token] / document_length for token in doc_word_count}
        alpha = self.parameters['alpha']
        mixed_prob = {token: (1 - alpha) * document_term_prob[token] + alpha * collection_term_probs[token] for token in doc_word_count}
        mu = 1 / self.parameters['lambda'] - 1

        # 3. For all query_parts, compute score
        doc_score = [
            query_parts_count[token] * math.log(1 + mu * doc_word_count[token] / mixed_prob[token] / document_length)
                for token in doc_word_count
        ]
        doc_score = sum(doc_score) if len(doc_score) != 0 else 0
        # 4. Return the score
        return {'docid': docid, 'score': doc_score}