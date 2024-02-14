"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization,
and your own ranker.
"""
import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import CrossEncoder
from indexing import InvertedIndex

from document_preprocessor import RegexTokenizer
import math
import torch

class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    # TODO: implement this class properly. This is responsible for returning a list of sorted relevant documents.
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str], 
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int,str]) -> None:
        """
        Initializes the state of the Ranker object.

        NOTE: Previous homeworks had you passing the class of the scorer to this function.
            This has been changed as it created a lot of confusion.
            You should now pass an instantiated RelevanceScorer to this function.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict
        nonsense_token = ""
        self.stopwords_filter = lambda x: nonsense_token if str.lower(x) in stopwords else x

    def query_info_helper(self, query_word_counts: dict[str, int]) -> tuple[Counter, set[int]]:
        relevant_docs = set()
        doc_term_counts = defaultdict(Counter)
        for token in query_word_counts:
            doc_ids = set()
            for doc_info in self.index.get_postings(token):
                doc_id = doc_info[0]
                term_freq = doc_info[1]
                doc_term_counts[doc_id][token] = term_freq
                doc_ids.add(doc_id)
            relevant_docs.update(set(doc_ids))
        return doc_term_counts, relevant_docs

    def query_rank_helper(self, query_word_counts: dict[str, int], doc_term_counts: Counter,
            relevant_docs: set[int]) -> list[tuple[int, float]]:
        results = []
        for docid in relevant_docs:
            doc_score = self.scorer.score(docid, doc_term_counts[docid], query_word_counts)
            results.append((docid, doc_score))
        results.sort(key=lambda x: (-x[1], x[0]))
        return results

    def query(self, query: str, pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
              pseudofeedback_beta=0.2) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number
                 of top-ranked documents to be used in the query,
            pseduofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseduofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query

        Returns:
            A list containing tuples of the documents (ids) and their relevance scores

        NOTE: We are standardizing the query output of Ranker to match with L2RRanker.query and VectorRanker.query
        The query function should return a sorted list of tuples where each tuple has the first element as the document ID
        and the second element as the score of the document after the ranking process.
        """
        # TODO: Tokenize the query and remove stopwords, if needed
        results = []
        query_parts = self.tokenize(query)
        query_parts = list(map(self.stopwords_filter, query_parts))
        query_word_counts = Counter(query_parts)
        # TODO (HW4): If the user has indicated we should use feedback,
        #  create the pseudo-document from the specified number of pseudo-relevant results.
        #  This document is the cumulative count of how many times all non-filtered words show up
        #  in the pseudo-relevant documents. See the equation in the write-up. Be sure to apply the same
        #  token filtering and normalization here to the pseudo-relevant documents.
        doc_term_counts, relevant_docs = self.query_info_helper(query_word_counts)
        results = self.query_rank_helper(query_word_counts, doc_term_counts, relevant_docs)

        # print("First rank:", results)
        if pseudofeedback_num_docs > 0 and len(results) > 0:
            # TODO (HW4): Get the top N documents from the results and create a pseudo-document
            #  using the equation in the write-up. Be sure to apply the same token filtering and
            #  normalization here to the pseudo-relevant documents.
            pseudo_docids = [result[0] for result in results[:pseudofeedback_num_docs]]
            # TODO (HW4): Combine the document word count for the pseudo-feedback with the query to create a new query
            # NOTE (HW4): Since you're using alpha and beta to weight the query and pseudofeedback doc, the counts
            #  will likely be *fractional* counts (not integers) which is ok and totally expected.
            pseudo_doc_counts = Counter()
            word_filter = lambda x: x if x in self.index.vocabulary else ""
            # tokenizer = RegexTokenizer(r'\w+')
            for docid in pseudo_docids:
                doc_text = self.raw_text_dict.get(docid, "")
                doc_tokens = self.tokenize(doc_text)
                doc_tokens = list(map(self.stopwords_filter, doc_tokens))
                doc_tokens = list(map(word_filter, doc_tokens))
                doc_count = Counter(doc_tokens)
                pseudo_doc_counts += doc_count
            # TODO (HW4): Combine the document word count for the pseudo-feedback with the query to create a new query
            # NOTE (HW4): Since you're using alpha and beta to weight the query and pseudofeedback doc, the counts
            #  will likely be *fractional* counts (not integers) which is ok and totally expected.
            tokens = set(query_word_counts.keys()).union(set(pseudo_doc_counts.keys()))
            updated_query = Counter()
            for token in tokens:
                updated_query[token] = pseudofeedback_alpha * query_word_counts[token] \
                    + pseudofeedback_beta * pseudo_doc_counts[token] / len(pseudo_docids)
            # TODO: Fetch a list of possible documents from the index and create a mapping from
            #  a document ID to a dictionary of the counts of the query terms in that document.
            #  You will pass the dictionary to the RelevanceScorer as input.
            # print("Updated query:", updated_query)
            doc_term_counts, relevant_docs = self.query_info_helper(updated_query)
            # TODO: Rank the documents using a RelevanceScorer
            results = self.query_rank_helper(updated_query, doc_term_counts, relevant_docs)
            # print("Second rank:", results)
        return results


class RelevanceScorer:
    """
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    """
    # TODO: Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25,
    #  PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index: InvertedIndex, parameters) -> None:
        raise NotImplementedError
    
    def get_tokens(self, doc_word_counts, query_word_counts, null_token=""):
        tokens = set(doc_word_counts.keys()).intersection(set(query_word_counts.keys())) - set([null_token])
        return tokens

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        TODO (HW4): Note that the `query_word_counts` is now a dictionary of words and their counts.
            This is changed from the previous homeworks.
        """
        raise NotImplementedError


# TODO: Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query
        tokens = set(doc_word_counts.keys()).intersection(set(query_word_counts.keys())) - set([""])
        query_word_counts = {token: query_word_counts[token] for token in tokens}
        doc_word_counts = {token: doc_word_counts[token] for token in tokens}
        score = 0
        for token in tokens:
            score += query_word_counts[token] * doc_word_counts[token]
        # 2. Return the score
        return score 


# TODO: Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        query_length = sum(query_word_counts.values()) if len(query_word_counts) != 0 else 0
        tokens = self.get_tokens(doc_word_counts, query_word_counts)
        query_word_counts = {token: query_word_counts[token] for token in tokens}
        doc_word_counts = {token: doc_word_counts[token] for token in tokens}
        # 2. Compute additional terms to use in algorithm
        doc_statistics = self.index.get_statistics()
        collection_length = doc_statistics.get('total_token_count', 0)
        total_term_freqs = {}
        for token in tokens:
            term_info = self.index.get_term_metadata(token)
            total_term_freqs[token] = term_info['term_freq'] if term_info is not None else 0
        collection_term_count = {token: total_term_freqs.get(token, 0) for token in tokens}
        document_length = self.index.get_doc_metadata(docid).get('length', 0)
        mu = self.parameters['mu']
        # 3. For all query_parts, compute score
        doc_score = [
            query_word_counts[token] * math.log(1 + (doc_word_counts[token] * collection_length / mu / collection_term_count[token]))
            for token in tokens
        ]
        score = sum(doc_score) if len(doc_score) != 0 else 0
        score += query_length * math.log(mu / (document_length + mu))
        # 4. Return the score
        return score
    
# TODO: Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index
        tokens = self.get_tokens(doc_word_counts, query_word_counts)        
        query_word_counts = {token: query_word_counts[token] for token in tokens}
        doc_word_counts = {token: doc_word_counts[token] for token in tokens}
        # 2. Compute additional terms to use in algorithm
        document_length = self.index.get_doc_metadata(docid).get('length', 0)
        doc_statistics = self.index.get_statistics()
        document_average_length = doc_statistics.get('mean_document_length', 0)
        number_of_documents = doc_statistics.get('number_of_documents', 0)
        document_appearance = {}
        for token in tokens:
            term_info = self.index.get_term_metadata(token)
            if term_info is not None:
                document_appearance[token] = term_info['doc_freq']
        doc_tf_down = 1 - self.b + self.b * (document_length / document_average_length)
        # 3. Find the dot product of the word count vector of the document and the word count vector of the query

        # 3. For all query parts, compute the TF and IDF to get a score
        doc_tf = {token: (self.k1 + 1) * doc_word_counts[token] / (self.k1 * doc_tf_down + doc_word_counts[token]) for token in tokens}
        doc_idf = {token: math.log((number_of_documents - document_appearance[token] + 0.5) / (document_appearance[token] + 0.5)) for token in tokens}
        qtf = {token: ((self.k3 + 1) * query_word_counts[token]) / (self.k3 + query_word_counts[token]) for token in tokens}
        doc_score = [doc_tf[token] * doc_idf[token] * qtf[token] for token in tokens]
        score = sum(doc_score) if len(doc_score) > 0 else 0
        # 4. Return score
        return score


# TODO (HW4): Implement Personalized BM25
class PersonalizedBM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, relevant_doc_index: InvertedIndex,
                 parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        """
        Initializes Personalized BM25 scorer.

        Args:
            index: The inverted index used to use for computing most of BM25
            relevant_doc_index: The inverted index of only documents a user has rated as relevant,
                which is used when calculating the personalized part of BM25
            parameters: The dictionary containing the parameter values for BM25

        Returns:
            The Personalized BM25 score
        """
        self.index = index
        self.relevant_doc_index = relevant_doc_index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # TODO (HW4): Implement Personalized BM25
        # 1. Get necessary information from index
        tokens = self.get_tokens(doc_word_counts, query_word_counts)        
        query_word_counts = {token: query_word_counts[token] for token in tokens}
        doc_word_counts = {token: doc_word_counts[token] for token in tokens}
        # 2. Compute additional terms to use in algorithm
        document_length = self.index.get_doc_metadata(docid).get('length', 0)
        doc_statistics = self.index.get_statistics()
        document_average_length = doc_statistics.get('mean_document_length', 0)
        num_doc = doc_statistics.get('number_of_documents', 0)
        num_seed_doc = self.relevant_doc_index.get_statistics().get('number_of_documents', 0)

        document_appearance = Counter()
        seed_doc_appearance = Counter()
        for token in tokens:
            term_info = self.index.get_term_metadata(token)
            if term_info is not None:
                document_appearance[token] = term_info['doc_freq']
            term_info = self.relevant_doc_index.get_term_metadata(token)
            if term_info is not None:
                seed_doc_appearance[token] = term_info['doc_freq']
        doc_tf_down = 1 - self.b + self.b * (document_length / document_average_length)
        # 3. For all query parts, compute the TF and IDF to get a score
        doc_tf = {token: (self.k1 + 1) * doc_word_counts[token] / (self.k1 * doc_tf_down + doc_word_counts[token]) for token in tokens}
        doc_idf = Counter()
        for token in tokens:
            r_w = seed_doc_appearance[token]
            df_w = document_appearance[token]
            doc_idf[token] = math.log((r_w + 0.5) / (df_w - r_w + 0.5) * (num_doc - df_w - num_seed_doc + r_w + 0.5) / (num_seed_doc - r_w + 0.5))

        qtf = {token: ((self.k3 + 1) * query_word_counts[token]) / (self.k3 + query_word_counts[token]) for token in tokens}
        doc_score = [doc_tf[token] * doc_idf[token] * qtf[token] for token in tokens]
        # print(docid, *zip(tokens, doc_score))
        score = sum(doc_score) if len(doc_score) > 0 else 0
        # 4. Return score
        return score


# TODO: Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        tokens = self.get_tokens(doc_word_counts, query_word_counts)
        query_word_counts = {token: query_word_counts[token] for token in tokens}
        doc_word_counts = {token: doc_word_counts[token] for token in tokens}
        # 2. Compute additional terms to use in algorithm
        document_length = self.index.get_doc_metadata(docid).get('length', 0)
        doc_statistics = self.index.get_statistics()
        document_average_length = doc_statistics.get('mean_document_length', 0)
        number_of_documents = doc_statistics.get('number_of_documents', 0)
        document_appearance = {}
        for token in tokens:
            term_info = self.index.get_term_metadata(token)
            if term_info is not None:
                document_appearance[token] = term_info['doc_freq']
        doc_tf_down = 1 - self.b + self.b * (document_length / document_average_length)
        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
        doc_tf = {token: (1 + math.log(1 + math.log(doc_word_counts[token]))) / doc_tf_down for token in tokens}
        doc_idf = {token: math.log((number_of_documents + 1) / document_appearance[token]) for token in tokens}
        doc_score = [query_word_counts[token] * doc_tf[token] * doc_idf[token] for token in tokens]
        score = sum(doc_score) if len(doc_score) > 0 else 0
        # 4. Return the score
        return score


# TODO: Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        tokens = self.get_tokens(doc_word_counts, query_word_counts)
        # query_word_counts = {token: query_word_counts[token] for token in tokens}
        doc_word_counts = {token: doc_word_counts[token] for token in tokens}
        # 2. Compute additional terms to use in algorithm
        number_of_documents = self.index.get_statistics().get('number_of_documents', 1)
        document_appearance = {}
        for token in doc_word_counts:
            term_info = self.index.get_term_metadata(token)
            if term_info is not None:
                document_appearance[token] = term_info['doc_freq']
        # 3. For all query parts, compute the TF and IDF to get a score
        doc_tf = {token: math.log(doc_word_counts[token] + 1) for token in tokens}
        doc_idf = {token: 1 + math.log(number_of_documents / document_appearance[token]) for token in tokens}
        doc_score = [doc_tf[token] * doc_idf[token] for token in tokens]
        score = sum(doc_score) if len(doc_score) > 0 else 0
        return score


class CrossEncoderScorer:
    def __init__(self, raw_text_dict: dict[int, str],
                 cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model

        NOTE 1: The CrossEncoderScorer class uses a pre-trained cross-encoder model
            from the Sentence Transformers package to score a given query-document pair.

        NOTE 2: This is not a RelevanceScorer object because the method signature for score() does not match,
            but it has the same intent, in practice.
        """
        # TODO: Save any new arguments that are needed as fields of this class
        self.cross_encoder = CrossEncoder(cross_encoder_model_name, max_length=512)
        self.raw_text_dict = raw_text_dict

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


# TODO: Implement your own ranker with proper heuristics
class YourRanker(RelevanceScorer):
    def __init__(self, index, parameters: dict = {'lambda': 0.9, 'alpha': 0.5}) -> None:
        self.index = index
        self.parameters = parameters
    
    def score(self, docid: int,  doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> dict[str, int]:
        # 1. Get necessary information from index
        tokens = self.get_tokens(doc_word_counts, query_word_counts)
        query_word_counts = {token: query_word_counts[token] for token in tokens}
        doc_word_counts = {token: doc_word_counts[token] for token in tokens}
        # 2. Compute additional terms to use in algorithm
        doc_statistics = self.index.get_statistics()
        collection_length = doc_statistics.get('total_token_count', 0)
        total_term_freqs = {}
        for token in tokens:
            term_info = self.index.get_term_metadata(token)
            total_term_freqs[token] = term_info['term_freq'] if term_info is not None else 0
        document_length = self.index.get_doc_metadata(docid).get('length', 0)
        collection_term_freqs = {token: document_length * total_term_freqs[token] / collection_length 
                for token in tokens}
        alpha = self.parameters['alpha']
        mixed_probs = {token: (1 + alpha) * (doc_word_counts[token]/collection_term_freqs[token]) - alpha for token in tokens}
        mu = 1 / self.parameters['lambda'] - 1

        # 3. For all query_parts, compute score
        doc_score = [
            query_word_counts[token] * math.log(1 + mu * max(mixed_probs[token], 0))
                for token in tokens
        ]
        score = sum(doc_score) if len(doc_score) != 0 else 0
        # 4. Return the score
        return score


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        # Print randomly ranked results
        return 10

