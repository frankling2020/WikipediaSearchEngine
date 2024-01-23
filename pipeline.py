from models import BaseSearchEngine, SearchResponse

# your library imports go here
from document_preprocessor import RegexTokenizer
from indexing import Indexer, IndexType, read_dataset
from ranker import Ranker, CrossEncoderScorer, BM25, DirichletLM
from relevance import run_relevance_tests
# from L2RRanker import L2RRanker
from l2r import L2RRanker, L2RFeatureExtractor
from vector_ranker import VectorRanker

import json
import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm
from collections import defaultdict


class SearchEngine(BaseSearchEngine):
    def __init__(self, index_name: str) -> None:
        # 1. Create a document tokenizer using document_preprocessor Tokenizers
        preprocessor = RegexTokenizer('\w+')
        
        # 2. Load stopwords, network data, categories, etc
        STOPWORD_PATH = 'stopwords.txt'
        DATASET_PATH = 'wikipedia_200k_dataset.jsonl.gz'
        EDGELIST_PATH = 'edgelist.csv.gz'
        NETWORK_STATS_PATH = 'network_stats.csv'
        DOC_CATEGORY_INFO_PATH = 'doc_category_info.json'
        RECOGNIZED_CATEGORY_PATH = 'recognized_categories.txt'
        DOC2QUERY_PATH = 'doc2query.csv'
        MAIN_INDEX = 'main_index_augmented'
        TITLE_INDEX = 'title_index'
        RELEVANCE_TRAIN_DATA = 'hw3_relevance.train.csv'
        ENCODED_DOCUMENT_EMBEDDINGS_NPY_DATA = 'wiki-200k-vecs.msmarco-MiniLM-L12-cos-v5.npy'
        DOCUMENT_ID_TEXT = 'document-ids.txt'

        # Load stopwords
        print('Loading stopwords...')
        stopwords = set()
        with open(STOPWORD_PATH, 'r', encoding='utf-8') as file:
            for stopword in file:
                stopwords.add(stopword.strip())
        print(f'Stopwords collected {len(stopwords)}')

        # Load categories
        print('Loading categories...')
        doc_category_info = {}
        with open(DOC_CATEGORY_INFO_PATH, 'r') as f:
            doc_category_info = json.load(f)
            doc_category_info = {int(k): v for k, v in doc_category_info.items()}
        recognized_categories = set()
        with open(RECOGNIZED_CATEGORY_PATH, 'r') as f:
            recognized_categories = set(map(lambda x: x.strip(), f.readlines()))
        
        # Load network features
        print('Loading network features...')
        networks_stats = pd.read_csv(NETWORK_STATS_PATH, index_col=0)
        network_features = {}
        for row in tqdm(networks_stats.iterrows()):
            network_features[row[1]['docid']] = row[1][1:].to_dict()
        print(f'Network stats collection {len(network_features)}')

        # Load document augmentations
        print('Loading document augmentations...')
        doc_augment_dict = defaultdict(lambda: [])
        # doc2query_df = pd.read_csv(DOC2QUERY_PATH).dropna()
        # for row in tqdm(doc2query_df.iterrows(), total=len(doc2query_df)):
        #     doc_id = int(row[1]['doc'])
        #     doc_query = row[1]['query']
        #     doc_augment_dict[doc_id].append(doc_query)
        # print(f'Document augmentations collected {len(doc_augment_dict)}')

        # Load raw documents (first 500 words)
        print("Loading raw documents...")
        # raw_text_dict = {}
        # for doc in tqdm(read_dataset(DATASET_PATH)):
        #     doc_id = int(doc['docid'])
        #     raw_text_dict[doc_id] = " ".join(preprocessor.tokenize(doc['text'])[:500])

        # Load document embeddings
        print('Loading document embeddings...')
        encoded_docs = None
        with open(ENCODED_DOCUMENT_EMBEDDINGS_NPY_DATA, 'rb') as file:
            encoded_docs = np.load(file)
        with open(DOCUMENT_ID_TEXT, 'r') as f:
            document_ids = f.read().splitlines()
            document_ids = [int(x) for x in document_ids]

        # 3. Create an index using the Indexer and IndexType (with the Wikipedia JSONL and stopwords)
        dataset_path = 'wikipedia_200k_dataset.jsonl.gz'
        print('Loading document text and title index...')
        t0 = time()
        main_index = Indexer.load_index(MAIN_INDEX)
        t1 = time()
        print('Done in %.2f seconds' % (t1- t0))
        title_index = Indexer.load_index(TITLE_INDEX)
        t2 = time()
        print('Done in %.2f seconds' % (t2- t1))

        # 4. Initialize a Ranker/L2RRanker with the index, stopwords, etc.
        # 5. If using L2RRanker, train it here.
        # cescorer = CrossEncoderScorer(raw_text_dict)
        cescorer = None
        self.feature_extractor = L2RFeatureExtractor(main_index, title_index, doc_category_info,
                                preprocessor, stopwords, recognized_categories,
                                network_features, cescorer)
        # self.base_ranker = VectorRanker('sentence-transformers/msmarco-MiniLM-L12-cos-v5', encoded_docs, document_ids)
        self.ranker = VectorRanker('sentence-transformers/msmarco-MiniLM-L12-cos-v5', encoded_docs, document_ids)
        # self.base_ranker = Ranker(main_index, preprocessor, stopwords, BM25(main_index))
        # self.ranker = L2RRanker(main_index, title_index, preprocessor,
                                # stopwords, self.base_ranker, self.feature_extractor)
        # print('Training ranker...')
        # t0 = time()
        # training_data_filename = 'hw3_relevance.train.csv'
        # self.ranker.train(training_data_filename)
        # self.ranker.model.save('bm25_cross_model.txt')
        # t1 = time()
        print('Done in %.2f seconds' % (t1- t0))
        print("Evaluating ranker...")
        dev_data_filename = 'hw3_relevance.dev.csv'
        dev_info = run_relevance_tests(dev_data_filename, self.ranker)
        test_data_filename = 'hw3_relevance.test.csv'
        test_info = run_relevance_tests(test_data_filename, self.ranker)
        print("Dev MAP:", dev_info['map'])
        print("Dev NDCG:", dev_info['ndcg'])
        print("Test MAP:", test_info['map'])
        print("Test NDCG:", test_info['ndcg'])

    def search(self, query: str) -> list[SearchResponse]:
        # 1. Use the ranker object to query the search pipeline
        # 2. This is example code and may not be correct.
        results = self.ranker.query(query)
        return [SearchResponse(id=idx+1, docid=result[0], score=result[1]) for idx, result in enumerate(results)]


def initialize():
    # search_obj = SearchEngine('test_index')
    search_obj = SearchEngine('BasicInvertedIndex')
    return search_obj
