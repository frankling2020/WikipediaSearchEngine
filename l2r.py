from tqdm import tqdm
import pandas as pd
import lightgbm
from indexing import InvertedIndex
import multiprocessing
from collections import defaultdict, Counter
import numpy as np
from document_preprocessor import Tokenizer
from ranker import Ranker, TF_IDF, BM25, PivotedNormalization, CrossEncoderScorer


# TODO: scorer has been replaced with ranker in initialization, check README for more details
class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor') -> None:
        """
        Initializes a L2RRanker system.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object
            feature_extractor: The L2RFeatureExtractor object
        """
        # TODO: Save any new arguments that are needed as fields of this class
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.ranker = ranker
        self.feature_extractor = feature_extractor
        # TODO: Initialize the LambdaMART model (but don't train it yet)
        self.model = LambdaMART()
                   
    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores: A dictionary of queries mapped to a list of
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            X (list): A list of feature vectors for each query-document pair
            y (list): A list of relevance scores for each query-document pair
            qgroups (list): A list of the number of documents retrieved for each query
        """
        # NOTE: qgroups is not the same length as X or y
        #       This is for LightGBM to know how many relevance scores we have per query
        X = []
        y = []
        qgroups = []

        # TODO: for each query and the documents that have been rated for relevance to that query,
        # process these query-document pairs into features
        for query, doc_rel_scores in tqdm(query_to_document_relevance_scores.items(), desc="Preparing"):
            cnt = 0
            # TODO: Accumulate the token counts for each document's title and content here
            query_parts = self.document_preprocessor.tokenize(query)
            doc_word_counts = L2RRanker.accumulate_doc_term_counts(self.document_index, query_parts)
            title_word_counts = L2RRanker.accumulate_doc_term_counts(self.title_index, query_parts)

            # TODO: For each of the documents, generate its features, then append
            # the features and relevance score to the lists to be returned
            for docid, rel_score in doc_rel_scores:
                feature = self.feature_extractor.generate_features(docid, doc_word_counts[docid], title_word_counts[docid], query_parts, query)
                X.append(feature)
                y.append(rel_score)
                cnt += 1
            # TODO: Make sure to keep track of how many scores we have for this query in qrels
            qgroups.append(cnt)
        return X, y, qgroups

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document

        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
        """
        # TODO: Retrieve the set of documents that have each query word (i.e., the postings) and
        #       create a dictionary that keeps track of their counts for the query word
        doc_term_counts = defaultdict(Counter)
        for token in query_parts:
            for doc_info in index.get_postings(token):
                doc_id = doc_info[0]
                term_freq = doc_info[1]
                doc_term_counts[doc_id][token] = term_freq
        return doc_term_counts

    def extract_data_helper(self, data_filename: str) -> dict[str, list[tuple[int, int]]]:
        """A helper function that reads the data file and returns a dictionary
            mapping each query to a list of documents and their relevance scores for that query.
        """
        df = pd.read_csv(data_filename)
        query_to_document_relevance_scores = df.groupby('query')[['docid', 'rel']] \
                .apply(lambda x: list(map(tuple, x.values))).to_dict()
        return query_to_document_relevance_scores

    def train(self, training_data_filename: str, evaluation_data_filename: str = None) -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data jsonl file.

        Args:
            training_data_filename (str): a filename for a jsonl file containing documents and relevance scores
        """
        # TODO: Convert the relevance data into the right format for training data preparation
        query_to_document_relevance_scores = self.extract_data_helper(training_data_filename)
        # TODO: prepare the training data by featurizing the query-doc pairs and
        # getting the necessary datastructures
        X, y, qgroups = self.prepare_training_data(query_to_document_relevance_scores)
        kwargs = {}
        ## DEBUG
        if evaluation_data_filename is not None:
            eval_query_to_document_relevance_scores = self.extract_data_helper(evaluation_data_filename)
            X_eval, y_eval, qgroups_eval = self.prepare_training_data(eval_query_to_document_relevance_scores)
            eval_set = [(np.array(X_eval), np.array(y_eval))]
            kwargs = {'eval_set': eval_set, 'eval_group': [qgroups_eval], 'eval_metric': 'ndcg', 'eval_at': [10]}
        # TODO: Train the model
        self.model.fit(X, y, qgroups, **kwargs)

    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        # TODO: Return a prediction made using the LambdaMART model
        return self.model.predict(X)
    
    def relevant_doc_filter(self, query_parts: list[str]) -> set[int]:
        relevant_docs = set()
        for token in set(query_parts):
            docids = [doc_info[0] for doc_info in self.document_index.get_postings(token)]
            relevant_docs.update(set(docids))
        return relevant_docs

    def query(self, query: str, **kwargs) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        # TODO: Retrieve potentially-relevant documents
        query_parts = self.document_preprocessor.tokenize(query)
        if len(query_parts) == 0:
            return []

        # relevant_docs = self.relevant_doc_filter(query_parts)
        # TODO: Fetch a list of possible documents from the index and create a mapping from
        #       a document ID to a dictionary of the counts of the query terms in that document.
        #       You will pass the dictionary to the RelevanceScorer as input
        #
        # NOTE: we collect these here (rather than calling a Ranker instance) because we'll
        #       pass these doc-term-counts to functions later, so we need the accumulated representations

        # TODO: Accumulate the documents word frequencies for the title and the main body
        doc_term_counts = L2RRanker.accumulate_doc_term_counts(self.document_index, query_parts)
        title_term_counts = L2RRanker.accumulate_doc_term_counts(self.title_index, query_parts)

        # TODO: Score and sort the documents by the provided scorer for just the document's main text (not the title).
        #       This ordering determines which documents we will try to *re-rank* using our L2R model
        naive_ranked_docs = self.ranker.query(query, **kwargs)

        naive_ranked_docs = [(docid, score) for docid, score in naive_ranked_docs if docid in doc_term_counts]
        # TODO: Filter to just the top 100 documents for the L2R part for re-ranking
        possible_docs = [docid for docid, _ in naive_ranked_docs[:100]]

        # TODO: Construct the feature vectors for each query-document pair in the top 100
        if len(possible_docs) > 0:
            docids = []
            feature_vecs = []
            for docid in possible_docs:
                feature_vec = self.feature_extractor.generate_features(docid, doc_term_counts.get(docid, {}), title_term_counts.get(docid, {}), query_parts, query)
                docids.append(docid)
                feature_vecs.append(feature_vec)

            # TODO: Use your L2R model to rank these top 100 documents
            preds = self.predict(feature_vecs)
            # TODO: Sort posting_lists based on scores
            ranked_docids = [(docid, -score) for score, docid in sorted(zip(map(lambda x: -x, preds), docids))]
            # TODO: Make sure to add back the other non-top-100 documents that weren't re-ranked        
            naive_ranked_docs = ranked_docids + naive_ranked_docs[100:]
        # TODO: Return the ranked documents
        return naive_ranked_docs


class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 doc_category_info: dict[int, list[str]],
                 document_preprocessor: Tokenizer, stopwords: set[str],
                 recognized_categories: set[str], docid_to_network_features: dict[int, dict[str, float]],
                 ce_scorer: CrossEncoderScorer) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "page_rank", "hub_score", and "authority_score"
                and values with the scores for those features
            ce_scorer: The CrossEncoderScorer object
        """
        # TODO: Set the initial state using the arguments
        self.document_index = document_index
        self.title_index = title_index
        self.doc_category_info = doc_category_info
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.docid_to_network_features = docid_to_network_features
        # TODO: For the recognized categories (i.e,. those that are going to be features), consider
        #       how you want to store them here for faster featurizing
        self.category_to_id = {k: v for v, k in enumerate(recognized_categories)}
        # TODO (HW2): Initialize any RelevanceScorer objects you need to support the methods below.
        #             Be sure to use the right InvertedIndex object when scoring
        self.bm25_doc = BM25(self.document_index)
        self.pivoted_norm_doc = PivotedNormalization(self.document_index)
        self.ce_scorer = ce_scorer

    # TODO: Article Length
    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        return self.document_index.get_doc_metadata(docid)['length']

    # TODO: Title Length
    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        return self.title_index.get_doc_metadata(docid)['length']

    # TODO: TF
    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """
        tokens = set(word_counts.keys()).intersection(set(query_parts))
        tf_doc = [np.log(1 + v) for k, v in word_counts.items() if k in tokens]
        return sum(tf_doc)

    # TODO: TF-IDF
    def get_tf_idf(self, index: InvertedIndex, docid: int,
                   word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        query_parts = Counter(query_parts)
        tokens = set(word_counts.keys()).intersection(set(query_parts.keys()))
        tf_idf_score = 0
        total_docs_num = index.get_statistics().get('number_of_documents', 1)
        for token in tokens:
            doc_appearance = index.get_term_metadata(token)['doc_freq']
            # tf_idf_score += np.log(1 + word_counts[token]) * total_docs_num / doc_appearance
            tf_idf_score += np.log(1 + word_counts[token]) * (1 + np.log(total_docs_num / doc_appearance))
        return tf_idf_score

    # TODO: BM25
    def get_BM25_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        # TODO: Calculate the BM25 score and return it
        return self.bm25_doc.score(docid, doc_word_counts, Counter(query_parts))

    # TODO: Pivoted Normalization
    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        # TODO: Calculate the pivoted normalization score and return it
        return self.pivoted_norm_doc.score(docid, doc_word_counts, Counter(query_parts))

    # TODO: Document Categories
    def get_document_categories(self, docid: int) -> list:
        """
        Generates a list of binary features indicating which of the recognized categories that the document has.
        Category features should be deterministically ordered so list[0] should always correspond to the same
        category. For example, if a document has one of the three categories, and that category is mapped to
        index 1, then the binary feature vector would look like [0, 1, 0].

        Args:
            docid: The id of the document

        Returns:
            A list containing binary list of which recognized categories that the given document has
        """
        doc_categories = [0 for _ in range(len(self.category_to_id))]
        for category in self.doc_category_info[docid]:
            if category in self.category_to_id:
                doc_categories[self.category_to_id[category]] = 1
        return doc_categories

    # TODO: PageRank
    def get_pagerank_score(self, docid: int) -> float:
        """
        Gets the PageRank score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The PageRank score
        """
        return self.docid_to_network_features.get(docid, {}).get('pagerank', 0)

    # TODO: HITS Hub
    def get_hits_hub_score(self, docid: int) -> float:
        """
        Gets the HITS hub score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS hub score
        """
        return self.docid_to_network_features.get(docid, {}).get('hub_score', 0)

    # TODO: HITS Authority
    def get_hits_authority_score(self, docid: int) -> float:
        """
        Gets the HITS authority score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS authority score
        """
        return self.docid_to_network_features.get(docid, {}).get('authority_score', 0)

    # TODO (HW3): Cross-Encoder Score
    def get_cross_encoder_score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The Cross-Encoder score
        """
        return self.ce_scorer.score(docid, query)

    # TODO: Add at least one new feature to be used with your L2R model
    def get_weighted_unigram(self, doc_word_counts: dict[str, int],
                        query_parts: list[str]) -> float:
        weighted_score = 0
        if len(query_parts) != 0:
            tokens = set(doc_word_counts.keys()).intersection(set(query_parts))
            sorted_word_counts = sorted([doc_word_counts[token] for token in tokens if doc_word_counts[token] > 0], reverse=True)
            num_tokens = len(sorted_word_counts)
            weighted_sum = 0
            if num_tokens > 0:
                weighted_sum = np.sum([sorted_word_counts[i] * np.log(i + 2) for i in range(len(sorted_word_counts))])
            weighted_score = weighted_sum / len(set(query_parts))
        return weighted_score

    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str],
                          query: str) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            title_word_counts: The words in the document's title mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """
        # NOTE: We can use this to get a stable ordering of features based on consistent insertion
        #       but it's probably faster to use a list to start

        feature_vector = []

        # TODO: Document Length
        feature_vector.append(self.get_article_length(docid))
        # TODO: Title Length
        feature_vector.append(self.get_title_length(docid))
        # TODO Query Length
        feature_vector.append(len(query_parts))
        # TODO: TF (document)
        feature_vector.append(self.get_tf(self.document_index, docid, doc_word_counts, query_parts))
        # TODO: TF-IDF (document)
        feature_vector.append(self.get_tf_idf(self.document_index, docid, doc_word_counts, query_parts))
        # TODO: TF (title)
        feature_vector.append(self.get_tf(self.title_index, docid, title_word_counts, query_parts))
        # TODO: TF-IDF (title)
        feature_vector.append(self.get_tf_idf(self.title_index, docid, title_word_counts, query_parts))
        # TODO: BM25
        feature_vector.append(self.get_BM25_score(docid, doc_word_counts, query_parts))
        # TODO: Pivoted Normalization
        feature_vector.append(self.get_pivoted_normalization_score(docid, doc_word_counts, query_parts))
        # TODO: Pagerank
        feature_vector.append(self.get_pagerank_score(docid))
        # TODO: HITS Hub
        feature_vector.append(self.get_hits_hub_score(docid))
        # TODO: HITS Authority
        feature_vector.append(self.get_hits_authority_score(docid))
        # TODO: Cross-Encoder Score
        if self.ce_scorer is not None:
            feature_vector.append(self.get_cross_encoder_score(docid, query))
        # TODO: Add at least one new feature to be used with your L2R model.
        feature_vector.append(self.get_weighted_unigram(doc_word_counts, query_parts))
        # TODO: Calculate the Document Categories features.
        # NOTE: This should be a list of binary values indicating which categories are present.        
        feature_vector.extend(self.get_document_categories(docid))
        return feature_vector


class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 20,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.005,
            'max_depth': -1,
            # NOTE: You might consider setting this parameter to a higher value equal to
            # the number of CPUs on your machine for faster training
            "n_jobs": multiprocessing.cpu_count()-1,
            "verbosity": 1,
        }

        if params:
            default_params.update(params)

        # TODO: Initialize the LGBMRanker with the provided parameters and assign as a field of this class
        self.model = lightgbm.LGBMRanker(**default_params)

    def fit(self, X_train, y_train, qgroups_train, **kwargs):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples
            y_train (array-like): Target values
            qgroups_train (array-like): Query group sizes for training data

        Returns:
            self: Returns the instance itself
        """
        # TODO: Fit the LGBMRanker's parameters using the provided features and labels
        self.model.fit(X_train, y_train, group=qgroups_train, **kwargs)
        return self

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like): 
                A list of featurized documents where each document is a list of its features
                All documents should have the same length

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """
        # TODO: Generate the predicted values using the LGBMRanker
        return self.model.predict(featurized_docs)
    
    def save(self, file_name: str = 'l2r.model.txt') -> None:
        self.model.booster_.save_model(file_name)
    
    def load(self, file_name: str = 'l2r.model.txt') -> None:
        self.model = lightgbm.Booster(model_file=file_name)

