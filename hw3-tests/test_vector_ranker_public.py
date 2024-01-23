import unittest
from vector_ranker import VectorRanker
from sentence_transformers import SentenceTransformer
import numpy as np
import json

from test_l2r_public import TestL2RRanker
from l2r import L2RRanker


def assertScoreLists(self, exp_list, res_list):
    self.assertEqual(len(exp_list), len(
        res_list), f'Expected length {len(exp_list)} but actual list length {len(res_list)}')
    for idx in range(len(res_list)):
        self.assertEqual(exp_list[idx][0], res_list[idx][0],
                         f'Expected document not at index {idx}')
        self.assertAlmostEqual(exp_list[idx][1], res_list[idx][1], places=4,
                               msg=f'Expected score differs from actual score at {idx}')


class TestVectorRanker(unittest.TestCase):
    def setUp(self) -> None:
        self.model_name = 'sentence-transformers/msmarco-MiniLM-L12-cos-v5'
        self.transformer = SentenceTransformer(self.model_name)
        self.doc_embeddings = []
        self.doc_ids = []
        with open('./toy_dataset.jsonl','r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                self.doc_embeddings.append(self.transformer.encode(data['text']))
                self.doc_ids.append(data['docid'])
        self.doc_embeddings = np.array(self.doc_embeddings)

    def test_query(self):
        exp_list = [(2, 0.509707510471344), (1, 0.38314512372016907), (3, 0.28278106451034546)]
        ranker = VectorRanker(self.model_name, self.doc_embeddings, self.doc_ids)
        query = 'What is the second document?'
        res_list = ranker.query(query)
        self.assertIsInstance(res_list, list)
        self.assertIsInstance(res_list[0], tuple)
        assertScoreLists(self, exp_list, res_list)
    
    def test_query_empty(self):
        exp_list = []
        ranker = VectorRanker(self.model_name, self.doc_embeddings, self.doc_ids)
        query = ''
        res_list = ranker.query(query)
        assertScoreLists(self, exp_list, res_list)
    
    def test_query_empty_docs(self):
        exp_list = []
        ranker = VectorRanker(self.model_name, np.array([]), [])
        query = 'What is the second document?'
        res_list = ranker.query(query)
        assertScoreLists(self, exp_list, res_list)


class TestL2RVectorRanker(TestL2RRanker):
    def setUp(self):
        super().setUp()
        self.model_name = 'sentence-transformers/msmarco-MiniLM-L12-cos-v5'
        self.transformer = SentenceTransformer(self.model_name)
        self.doc_embeddings = []
        self.doc_ids = []
        with open('./data.jsonl','r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                self.doc_embeddings.append(self.transformer.encode(data['text']))
                self.doc_ids.append(data['docid'])
        self.doc_embeddings = np.array(self.doc_embeddings)
        self.ranker = VectorRanker(self.model_name, self.doc_embeddings, self.doc_ids)
    
    # @unittest.skip("Test broken")
    # def test_query_with_noexistient_term(self):
        # pass


if __name__ == '__main__':
    unittest.main()
