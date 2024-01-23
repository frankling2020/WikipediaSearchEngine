import unittest
import json
from collections import Counter, defaultdict

from ranker import Ranker, WordCountCosineSimilarity, DirichletLM, BM25, PersonalizedBM25, PivotedNormalization, TF_IDF
from indexing import Indexer, IndexType
from document_preprocessor import RegexTokenizer


def assertScoreLists(self, exp_list, res_list):
    self.assertEqual(len(exp_list), len(
        res_list), f'Expected length {len(exp_list)} but actual list length {len(res_list)}')
    for idx in range(len(res_list)):
        self.assertEqual(exp_list[idx][0], res_list[idx][0],
                         f'Expected document not at index {idx}')
        self.assertAlmostEqual(exp_list[idx][1], res_list[idx][1], places=4,
                               msg=f'Expected score differs from actual score at {idx}')

def get_raw_text_dict(self, dataset_name):
    raw_text_dict = {}
    with open(dataset_name) as f:
        for line in f:
            d = json.loads(line)
            docid = d['docid']
            tokens = (d['text'])
            raw_text_dict[docid] = tokens

    return raw_text_dict


class MockTokenizer:
    def tokenize(self, text):
        return text.split()

class TestScoreBase(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = MockTokenizer()
        self.stopwords = set(['a', 'an', 'and', 'the', 'this'])
        self.index = Indexer.create_index(
            IndexType.InvertedIndex, './data.jsonl', self.preprocessor, self.stopwords, 1)
        self.doc_word_counts = {'to': 2, 'AI': 1, 'ML': 1, 
            'algorithms': 1, 'drive': 1, 'personal': 1, 
            'assistants': 1, 'like': 1, 'Siri,': 1, 'Alexa,': 1, 
            'Google': 1, 'Assistant.': 1, 'These': 1, 'digital': 1, 
            'companions': 1, 'understand': 1, 'respond': 1, 'natural': 1, 
            'language,': 1, 'making': 1, 'them': 1, 'invaluable': 1, 
            'for': 1, 'tasks': 1, 'ranging': 1, 'from': 1, 'setting': 1, 
            'reminders': 1, 'answering': 1, 'questions.': 1
        }
        self.relevant_doc_index = Indexer.create_index(
            IndexType.InvertedIndex, './data_relevant.jsonl', self.preprocessor, self.stopwords, 1)
        self.raw_text_dict = self.get_raw_text_dict('data.jsonl')

    def get_raw_text_dict(self, dataset_name):
        raw_text_dict = {}
        with open(dataset_name) as f:
            for line in f:
                d = json.loads(line)
                docid = d['docid']
                tokens = (d['text'])
                raw_text_dict[docid] = tokens
        return raw_text_dict


class TestWordCountCosineSimilarity(TestScoreBase):
    def setUp(self) -> None:
        super().setUp()
        scorer = WordCountCosineSimilarity(self.index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer, self.raw_text_dict)

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        self.assertEqual(exp_list, res_list,
                         'Cosine: no overlap between query and docs')

    def test_perfect_match(self):
        exp_list = [(1, 1), (3, 1), (5, 1)]
        res_list = self.ranker.query("AI")
        self.assertEqual(exp_list, res_list,
                         'Expected list differs from result list')

    def test_partial_match(self):
        exp_list = [(3, 2), (4, 2), (1, 1), (5, 1)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        self.assertEqual(exp_list, res_list,
                         'Expected list differs from result list')


class TestDirichletLM(TestScoreBase):
    def setUp(self) -> None:
        super().setUp()
        scorer = DirichletLM(self.index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer, self.raw_text_dict)

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match(self):
        exp_list = [(5, 0.01128846343027107), (3, 0.007839334610553066),
                    (1, 0.0073475716303944075)]
        res_list = self.ranker.query("AI")
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match(self):
        exp_list = [(3, 0.029667610688458967), (4, 0.017285590697028078),
                    (5, -0.027460212369367794), (1, -0.04322377956887445)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        assertScoreLists(self, exp_list, res_list)

    def test_small_mu(self):
        DLM = DirichletLM(self.index, {'mu': 5})
        query = ['AI', 'Google']
        ret_score = DLM.score(1, self.doc_word_counts, Counter(query))
        exp_score = 1.6857412751512575

        self.assertAlmostEqual(
            exp_score, ret_score, places=3, msg='DirichletLM: partial match, score')

    def test_small_mu2(self):
        DLM = DirichletLM(self.index, {'mu': 1})
        query = ['AI', 'Google']
        ret_score = DLM.score(1, self.doc_word_counts, Counter(query))
        exp_score = 1.798539156213434

        self.assertAlmostEqual(
            exp_score, ret_score, places=3, msg='DirichletLM: partial match, score')

    def test_med_mu(self):
        DLM = DirichletLM(self.index, {'mu': 30})
        query = ['AI', 'Google']
        ret_score = DLM.score(1, self.doc_word_counts, Counter(query))
        exp_score = 1.2278314183215069

        self.assertAlmostEqual(
            exp_score, ret_score, places=3, msg='DirichletLM: partial match, score')

    def test_large_mu(self):
        DLM = DirichletLM(self.index, {'mu': 1000})
        query = ['AI', 'Google']
        ret_score = DLM.score(1, self.doc_word_counts, Counter(query))
        exp_score = 0.11811761538891903

        self.assertAlmostEqual(
            exp_score, ret_score, places=3, msg='DirichletLM: partial match, score')


class TestBM25(TestScoreBase):
    def setUp(self) -> None:
        super().setUp()
        scorer = BM25(self.index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer, self.raw_text_dict)

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match(self):
        exp_list = [(1, -0.31623109945742595), (3, -0.32042144088133173),
                    (5, -0.35318117923823517)]
        res_list = self.ranker.query("AI")
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match(self):
        exp_list = [(4, 1.5460888344441546), (3, 0.7257835477973098),
                    (1, -0.31623109945742595), (5, -0.35318117923823517)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        assertScoreLists(self, exp_list, res_list)

    def test_small_k1(self):
        scorer = BM25(self.index, {'b': 0.75, 'k1': 1, 'k3': 8})
        query = ['AI', 'Google']
        ret_score = scorer.score(1, self.doc_word_counts, Counter(query))
        exp_score = 0.7199009648250208

        self.assertAlmostEqual(exp_score, ret_score,
                               places=3, msg='BM25: partial match, score')

    def test_large_k1(self):
        scorer = BM25(self.index, {'b': 0.75, 'k1': 2, 'k3': 8})
        query = ['AI', 'Google']
        ret_score = scorer.score(1, self.doc_word_counts, Counter(query))
        exp_score = 0.7068428242958602

        self.assertAlmostEqual(exp_score, ret_score,
                               places=3, msg='BM25: partial match, score')

    def test_small_k3(self):
        scorer = BM25(self.index, {'b': 0.75, 'k1': 1.2, 'k3': 0})
        query = ['AI', 'Google']
        ret_score = scorer.score(1, self.doc_word_counts, Counter(query))
        exp_score = 0.7162920454285571

        self.assertAlmostEqual(exp_score, ret_score,
                               places=3, msg='BM25: partial match, score')

    def test_large_k3(self):
        scorer = BM25(self.index, {'b': 0.75, 'k1': 1.2, 'k3': 1000})
        query = ['AI', 'Google']
        ret_score = scorer.score(1, self.doc_word_counts, Counter(query))
        exp_score = 0.7162920454285571

        self.assertAlmostEqual(exp_score, ret_score,
                               places=3, msg='BM25: partial match, score')

    def test_random_param(self):
        scorer = BM25(self.index, {'b': 0.75, 'k1': 1.99, 'k3': 49})
        query = ['AI', 'Google']
        ret_score = scorer.score(1, self.doc_word_counts, Counter(query))
        exp_score = 0.7069285957828516

        self.assertAlmostEqual(exp_score, ret_score,
                               places=3, msg='BM25: partial match, score')


class TestPivotedNormalization(TestScoreBase):
    def setUp(self) -> None:
        super().setUp()
        scorer = PivotedNormalization(self.index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer, self.raw_text_dict)

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match(self):
        exp_list = [(5, 0.7095587433308632), (3, 0.6765779252477553),
                    (1, 0.6721150101735617)]
        res_list = self.ranker.query("AI")
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match(self):
        exp_list = [(4, 2.7806792016468633), (3, 2.4255064908289246),
                    (5, 0.7095587433308632), (1, 0.6721150101735617)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        assertScoreLists(self, exp_list, res_list)

    def test_small_param(self):
        scorer = PivotedNormalization(self.index, {'b': 0})
        query = ['AI', 'Google']
        ret_score = scorer.score(1, self.doc_word_counts, Counter(query))
        exp_score = 2.4849066497880004

        self.assertAlmostEqual(exp_score, ret_score, places=3,
                               msg='PivotedNormalization: partial match, score')

    def test_large_param(self):
        scorer = PivotedNormalization(self.index, {'b': 1})
        query = ['AI', 'Google']
        ret_score = scorer.score(1, self.doc_word_counts, Counter(query))
        exp_score = 2.1487133971696237

        self.assertAlmostEqual(exp_score, ret_score, places=3,
                               msg='PivotedNormalization: partial match, score')


class TestTF_IDF(TestScoreBase):
    def setUp(self) -> None:
        super().setUp()
        scorer = TF_IDF(self.index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer, self.raw_text_dict)

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match(self):
        exp_list = [(1, 1.047224521431117),
                    (3, 1.047224521431117), (5, 1.047224521431117)]
        res_list = self.ranker.query("AI")
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match(self):
        exp_list = [(4, 2.866760557116562), (3, 2.8559490532810434),
                    (1, 1.047224521431117), (5, 1.047224521431117)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        assertScoreLists(self, exp_list, res_list)


class TestPersonalizedBM25(TestScoreBase):
    def setUp(self) -> None:
        super().setUp()
        scorer = PersonalizedBM25(self.index, self.relevant_doc_index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer, self.raw_text_dict)

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match(self):
        exp_list = [(5, 0.5361928163775443), (3, 0.48645761697856715), (1, 0.4800959219003819)]
        res_list = self.ranker.query("AI")
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match(self):
        exp_list = [(4, 1.5460888344441546), (3, 1.5326626056572086), (5, 0.5361928163775443), (1, 0.4800959219003819)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        assertScoreLists(self, exp_list, res_list)


    def test_partial_match_large_alpha_one_doc(self):
        exp_list = [(4, 3.966307269579602), (3, 1.8251400992267406), (1, 0.9431149048525114), (5, 0.4879957092874279), (2, 0.16496279785262208)]
        res_list = self.ranker.query("AI chatbots and vehicles", 1, 0.9, 0.1)
        print(res_list)
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match_large_alpha_all_docs(self):
        exp_list = [(4, 4.500943213942879), (3, 4.040853135452582), (1, 1.272973331547797), (2, -0.15703366700769897), (5, -3.7342779593271835)]
        res_list = self.ranker.query("AI chatbots and vehicles", 3, 0.9, 0.1)
        print(res_list)
        assertScoreLists(self, exp_list, res_list)

    def test_no_overlap_large_alpha_one_doc(self):
        exp_list = []
        res_list = self.ranker.query("cough drops", 1, 0.9, 0.1)
        print(res_list)
        assertScoreLists(self, exp_list, res_list)
    
    def test_empty_query_large_alpha_one_doc(self):
        exp_list = []
        res_list = self.ranker.query("", 1, 0.9, 0.1)
        print(res_list)
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match_large_alpha_one_doc(self):
        exp_list = [(1, 0.4369412322913588), (3, 0.272435244536684), (2, -0.24372404478819562), (5, -3.551149361590311)]
        res_list = self.ranker.query("AI", 1, 0.9, 0.1)
        print(res_list)
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match_large_alpha_all_docs(self):
        exp_list = [(1, 3.4514903689703624), (3, 2.9434801941939686), (4, 1.0885822762329107), (2, -0.09831432147349785), (5, -3.735678534809934)]
        res_list = self.ranker.query("AI", 3, 0.9, 0.1)
        print(res_list)
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match_large_beta_one_doc(self):
        exp_list = [(4, 21.370125230104176), (1, 4.199418230615301), (3, 3.694436809153018), (2, 1.3512121307254097), (5, 0.05957697959750493)]
        res_list = self.ranker.query("AI chatbots and vehicles", 1, 0.1, 0.9)
        print(res_list)
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match_large_beta_all_docs(self):
        exp_list = [(4, 25.03436135509248), (3, 21.663883987196336), (1, 6.460170011283814), (2, -1.2601160487220455), (5, -34.2802147504142)]
        res_list = self.ranker.query("AI chatbots and vehicles", 3, 0.1, 0.9)
        print(res_list)
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match_large_beta_one_doc(self):
        exp_list = [(1, 0.05334399132226464), (3, -1.340844334098642), (2, -1.9963463893325233), (5, -33.025105679614526)]
        res_list = self.ranker.query("AI", 1, 0.1, 0.9)
        print(res_list)
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match_large_beta_all_docs(self):
        exp_list = [(1, 24.254696075758087), (3, 20.37035467492644), (4, 8.519601245527406), (2, -0.7791452296834769), (5, -34.34722922889731)]
        res_list = self.ranker.query("AI", 3, 0.1, 0.9)
        print(res_list)
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match_large_beta_too_much_docs(self):
        exp_list = [(1, 24.254696075758087), (3, 20.37035467492644), (4, 8.519601245527406), (2, -0.7791452296834769), (5, -34.34722922889731)]
        res_list = self.ranker.query("AI", 5, 0.1, 0.9)
        print(res_list)
        assertScoreLists(self, exp_list, res_list)

if __name__ == '__main__':
    unittest.main()
