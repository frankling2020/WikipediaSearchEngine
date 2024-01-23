from models import BaseSearchEngine, SearchResponse

# your library imports go here
from document_preprocessor import RegexTokenizer
from indexing import Indexer, IndexType
from ranker import Ranker
# from L2RRanker import L2RRanker
from l2r import L2RRanker, L2RFeatureExtractor
from ranker import DirichletLM

import json
import pandas as pd
from time import time


class SearchEngine(BaseSearchEngine):
    def __init__(self, index_name: str) -> None:
        # 1. Create a document tokenizer using document_preprocessor Tokenizers
        self.document_preprocessor = RegexTokenizer('\w+')
        # 2. Load stopwords, network data, categories, etc
        stopwords_filename = 'stopwords.txt'
        doc_category_info_filename = 'doc_category_info.json'
        network_features_filename = 'network_features.csv'
        recognized_categories_filename = 'recognized_categories.txt'
        # Load stopwords
        print('Loading stopwords...')
        self.stopwords = set()
        with open(stopwords_filename, 'r') as f:
            self.stopwords = set(map(lambda x: x.strip(), f.readlines()))
        # Load categories
        print('Loading categories...')
        self.recognized_categories = set()
        with open(recognized_categories_filename, 'r') as f:
            self.recognized_categories = set(map(lambda x: x.strip(), f.readlines()))
        self.doc_category_info = {}
        with open(doc_category_info_filename, 'r') as f:
            self.doc_category_info = json.load(f)
            self.doc_category_info = {int(k): v for k, v in self.doc_category_info.items()}
        # Load network features
        print('Loading network features...')
        self.network_features = {}
        networks_stats = pd.read_csv('network_stats.csv', index_col=0)
        for row in networks_stats.iterrows():
            self.network_features[row[1]['docid']] = row[1][1:].to_dict()
        # 3. Create an index using the Indexer and IndexType (with the Wikipedia JSONL and stopwords)
        dataset_path = 'wikipedia_200k_dataset.jsonl.gz'

        # self.index = Indexer.create_index(
        #     index_type=IndexType.InvertedIndex,
        #     dataset_path=dataset_path,
        #     document_preprocessor=self.document_processor,
        #     stopwords=self.stopwords,
        #     minimum_word_frequency=50
        # )
        print('Loading document text and title index...')
        t0 = time()
        self.document_indexer = Indexer.load_index(index_name + '_doc')
        t1 = time()
        print('Done in %.2f seconds' % (t1- t0))
        self.title_indexer = Indexer.load_index(index_name + '_title')
        t2 = time()
        print('Done in %.2f seconds' % (t2- t1))

        # 4. Initialize a Ranker/L2RRanker with the index, stopwords, etc.
        # 5. If using L2RRanker, train it here.
        # self.ranker = Ranker(self.index, self.document_preprocessor, self.stopwords, DirichletLM)
        self.feature_extractor = L2RFeatureExtractor(
            self.document_indexer, 
            self.title_indexer, 
            self.doc_category_info, 
            self.document_preprocessor, 
            self.stopwords, 
            self.recognized_categories, 
            self.network_features
        )
        self.scorer = DirichletLM(self.document_indexer)
        self.ranker = L2RRanker(
            self.document_indexer, 
            self.title_indexer, 
            self.document_preprocessor, 
            self.stopwords, 
            self.scorer, 
            self.feature_extractor
        )
        print('Training ranker...')
        t0 = time()
        training_data_filename = 'hw2_relevance.train.csv'
        self.ranker.train(training_data_filename)
        t1 = time()
        print('Done in %.2f seconds' % (t1- t0))

    def search(self, query: str) -> list[SearchResponse]:
        # 1. Use the ranker object to query the search pipeline
        # 2. This is example code and may not be correct.
        results = self.ranker.query(query)
        # return [SearchResponse(id=idx+1, docid=result['docid'], score=result['score']) for idx, result in enumerate(results)]
        return [SearchResponse(id=idx+1, docid=result, score=1/(1+idx)) for idx, result in enumerate(results)]


def initialize():
    # search_obj = SearchEngine('test_index')
    search_obj = SearchEngine('BasicInvertedIndex')
    return search_obj
