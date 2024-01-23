'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
Use libraries such as tqdm, orjson, collections.Counter, shelve if you need them.
DO NOT use the pickle module.
'''
from enum import Enum
import json
import shelve
from collections import Counter, defaultdict
import os
from document_preprocessor import read_jsonl
from tqdm import tqdm
from bisect import bisect_left, insort_left
from sample_data import SAMPLE_DOCS


class IndexType(Enum):
    # the three types of index currently supported are InvertedIndex, PositionalIndex and OnDiskInvertedIndex
    PositionalIndex = 'PositionalIndex'
    InvertedIndex = 'InvertedIndex'
    OnDiskInvertedIndex = 'OnDiskInvertedIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    '''
    The base interface representing the data structure for all index classes.
    The functions are meant to be implemented in the actual index classes and not as part of this interface.
    '''

    def __init__(self, index_name) -> None:
        self.index_name = index_name  # name of the index
        self.statistics = {'vocab': {}, 'mean_document_length': 0, 'number_of_documents': 0, 'total_token_count': 0, 'unique_token_count': 0, 'total_token_count': 0}  # the central statistics of the index
        self.index = {}  # the index
        self.vocabulary = set()  # the vocabulary of the collection
        # metadata like length, number of unique tokens of the documents
        self.document_metadata = {}
        # OPTIONAL if using SPIMI, use this variable to keep track of the index segments.
        self.index_segment = 0
        self.doc_counter = 0
        # self.fake_id_to_docid = {}
        # self.docid_to_fake_id = {}


    # NOTE: The following functions have to be implemented in the three inherited classes and not in this class
    

    def remove_doc(self, docid: int) -> None:
        # TODO implement this to remove a document from the entire index and statistics
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        # TODO implement this to add documents to the index
        raise NotImplementedError

    def get_postings(self, term: str) -> list:
        # TODO implement this to fetch a term's postings from the index
        raise NotImplementedError
    
    def get_doc_metadata(self, docid: int) -> dict[str, int]:
        # TODO implement to fetch a particular documents stored metadata
        raise NotImplementedError
    
    def get_term_metadata(self, term: str) -> dict[str, int]:
        # TODO implement to fetch a particular terms stored metadata
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        # TODO calculate statistics like 'unique_token_count', 'total_token_count', 'number_of_documents', 'mean_document_length' and any other relevant central statistic.
        raise NotImplementedError

    def save(self) -> None:
        # TODO save the index files to disk
        raise NotImplementedError

    def load(self) -> None:
        # TODO load the index files from disk to a Python object
        raise NotImplementedError

    def flush_to_disk(self) -> None:
        # OPTIONAL TODO flush index segments created using SPIMI strategy to disk and increment the segment number
        raise NotImplementedError
    
    def calculate_stats(self, doc_inc: int, token_inc: int) -> None:
        # TO-DO: not use self.index
        self.statistics['unique_token_count'] = len(self.vocabulary)
        token_count = self.statistics['total_token_count'] + token_inc
        self.statistics['total_token_count'] = max(token_count, 0)
        doc_count = self.statistics['number_of_documents'] + doc_inc
        self.statistics['number_of_documents'] = max(doc_count, 0)
        if self.statistics['number_of_documents'] == 0:
            self.statistics['mean_document_length'] = 0
        else:
            self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents']

    def create_dir_if_not_exist(self) -> None:
        if not os.path.exists(f'{self.index_name}'):
            os.mkdir(f'{self.index_name}')
    
    def search_doc_pos(self, docid: int, token: str) -> int:
        """Search internal docid"""
        # use the binary search to find the position of the token in the document
        # return -1 if not found
        # assume the token is in the docid
        docs_info = self.index[token]
        if docid in self.document_metadata:
            map_docid_to_id = lambda docid: self.document_metadata[docid]['fake_id']
            pos = bisect_left(docs_info, map_docid_to_id(docid), key=lambda x: map_docid_to_id(x[0]))
            if pos != len(docs_info) and docs_info[pos][0] == docid:
                return pos
        return -1

    def remove_token_from_document(self, token: str, docid: int, on_disk: bool = False) -> None:
        if token in self.index:
            pos = self.search_doc_pos(docid, token)
            if pos != -1:
                # statistics
                token_count = self.index[token][pos][1]
                self.statistics['vocab'][token] -= token_count
                if self.statistics['vocab'][token] <= 0:
                    self.statistics['vocab'].pop(token)
                # index
                if on_disk:
                    tmp = self.index[token]
                    tmp.pop(pos)
                    self.index[token] = tmp
                else:
                    self.index[token].pop(pos)
                if len(self.index[token]) == 0:
                    self.index.pop(token)
                    self.vocabulary.remove(token)

    def add_token_to_document(self, token: str, docid: int, count: int, on_disk: bool = False) -> None:
        # index
        if token not in self.index:
            self.index[token] = []
            self.vocabulary.add(token)
        # if self.search_doc_pos(docid, token) == -1:
        #     if on_disk:
        #         tmp = self.index[token]
        #         insort_left(tmp, [docid, count])
        #         self.index[token] = tmp
        #     else:
        #         insort_left(self.index[token], [docid, count])
        if on_disk:
            tmp = self.index[token]
            tmp.append([docid, count])
            self.index[token] = tmp
        else:
            self.index[token].append([docid, count])
        # statistics
        if token not in self.statistics['vocab']:
            self.statistics['vocab'][token] = 0
        self.statistics['vocab'][token] += count

    def add_token_to_pos_doc(self, token: str, docid: int, positions: list) -> None:
        # index
        if token not in self.index:
            self.index.update({token: []})
            self.vocabulary.add(token)
        # pos = self.search_doc_pos(docid, token)
        # if pos == -1:
            # insort_left(self.index[token], [docid, len(positions), positions])
        self.index[token].append([docid, len(positions), positions])
        # statistics
        if token not in self.statistics['vocab']:
            self.statistics['vocab'][token] = 0
        self.statistics['vocab'][token] += 1

    def save_except_index(self) -> None:
        # in practice spimi will treat metadata
        with open(f'{self.index_name}/{self.index_name}_metadata.json', 'w') as f:
            json.dump(self.document_metadata, f)
        with open(f'{self.index_name}/{self.index_name}_statistics.json', 'w') as f:
            json.dump(self.statistics, f)
        with open(f'{self.index_name}/{self.index_name}_vocabulary.json', 'w') as f:
            json.dump(list(self.vocabulary), f)
    
    def load_except_index(self) -> None:
        # spimi will treat metadata
        with open(f'{self.index_name}/{self.index_name}_metadata.json', 'r') as f:
            self.document_metadata = json.load(f)
            # convert str to int
            self.document_metadata = {int(k): v for k, v in self.document_metadata.items()}
            fake_ids = [v['fake_id'] for v in self.document_metadata.values()]
            self.doc_counter = max(fake_ids) + 1
        with open(f'{self.index_name}/{self.index_name}_statistics.json', 'r') as f:
            self.statistics = json.load(f)
        with open(f'{self.index_name}/{self.index_name}_vocabulary.json', 'r') as f:
            self.vocabulary = set(json.load(f))
    
    def add_doc_to_fake(self, docid: int) -> int:
        fake_id = self.doc_counter
        self.doc_counter += 1
        return fake_id


class BasicInvertedIndex(InvertedIndex):
    def __init__(self, index_name) -> None:
        super().__init__(index_name)
        self.statistics['index_type'] = 'BasicInvertedIndex'
    # TODO implement all the functions mentioned in the interface
    # This is the typical inverted index where each term keeps track of documents and the term count per document.

    def remove_doc(self, docid: int) -> None:
        try:
            for token in self.vocabulary.copy():
                self.remove_token_from_document(token, docid)
            token_inc = self.document_metadata[docid]['length']
            del self.document_metadata[docid]
            self.calculate_stats(-1, -token_inc)
        except:
            raise KeyError(f'{docid} not in index')

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        token_counts = Counter(tokens)
        self.document_metadata[docid] = {
            'length': len(tokens), 
            'doc_token_count': len(set(tokens) - set([""])),
            'fake_id': self.add_doc_to_fake(docid),
        }
        for token, count in token_counts.items():
            if token != "":
                self.add_token_to_document(token, docid, count)
        self.calculate_stats(1, len(tokens))
    
    def get_postings(self, term: str) -> list:
        return self.index.get(term, [])
    
    def get_doc_metadata(self, docid: int) -> dict[str, int]:
        return self.document_metadata.get(docid, {})

    def get_term_metadata(self, term: str) -> dict[str, int]:
        return {
            "doc_freq": len(self.index.get(term, {})),
            "term_freq": self.statistics['vocab'].get(term, 0),
        } if term in self.vocabulary else None
    
    def get_statistics(self) -> dict[str, int]:
        return self.statistics
    
    def save(self) -> None:
        self.create_dir_if_not_exist()
        with open(f'{self.index_name}/{self.index_name}_index.json', 'w') as f:
            json.dump(self.index, f)
        self.save_except_index()

    def load(self) -> None:
        self.create_dir_if_not_exist()
        with open(f'{self.index_name}/{self.index_name}_index.json', 'r') as f:
            self.index = json.load(f)
        self.load_except_index()


class PositionalInvertedIndex(InvertedIndex):
    def __init__(self, index_name) -> None:
        super().__init__(index_name)
        self.statistics['index_type'] = 'PositionalInvertedIndex'
    # TODO implement all the functions mentioned in the interface
    # This is the positional inverted index where each term keeps track of documents and positions of the terms occring in the document.

    def remove_doc(self, docid: int) -> None:
        try:
            # copy a self.vocab to avoid RuntimeError: dictionary changed size during iteration
            for token in self.vocabulary.copy():
                self.remove_token_from_document(token, docid)
            token_inc = self.document_metadata[docid]['length']
            del self.document_metadata[docid]
            self.calculate_stats(-1, -token_inc)
        except:
            raise KeyError(f'{docid} not in index')

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        self.document_metadata[docid] = {
            'length': len(tokens), 
            'doc_token_count': len(set(tokens) - set([""])),
            'fake_id': self.add_doc_to_fake(docid),
        }
        token_pos = {}
        for i, token in enumerate(tokens):
            if token != "":
                if token not in token_pos:
                    token_pos[token] = []
                token_pos[token].append(i)
        for token, positions in token_pos.items():
            self.add_token_to_pos_doc(token, docid, positions)
        self.calculate_stats(1, len(tokens))
    
    def get_postings(self, term: str) -> list:
        return self.index.get(term, [])
        
    def get_doc_metadata(self, docid: int) -> dict[str, int]:
        return self.document_metadata.get(docid, {})
    
    def get_term_metadata(self, term: str) -> dict[str, int]:
        return {
            "doc_freq": len(self.index.get(term, {})),
            "term_freq": self.statistics['vocab'].get(term, 0),
        } if term in self.vocabulary else None
    
    def get_statistics(self) -> dict[str, int]:
        return self.statistics
    
    def save(self) -> None:
        self.create_dir_if_not_exist()
        with open(f'{self.index_name}/{self.index_name}_index.json', 'w') as f:
            json.dump(self.index, f)
        self.save_except_index()

    def load(self) -> None:
        self.create_dir_if_not_exist()
        with open(f'{self.index_name}/{self.index_name}_index.json', 'r') as f:
            self.index = json.load(f)
        self.load_except_index()
        

class OnDiskInvertedIndex(InvertedIndex):
    def __init__(self, index_name) -> None:
        super().__init__(index_name)
        self.statistics['index_type'] = 'OnDiskInvertedIndex'
        self.create_dir_if_not_exist()
        self.index = shelve.open(f'{self.index_name}/{self.index_name}', flag='c', writeback=True)
    # TODO implement all the functions mentioned in the interface
    # This is a typical inverted index which will be using Python's shelve module to persist and even read the data from the disk.

    def remove_doc(self, docid: int) -> None:
        try:
            for token in self.vocabulary.copy():
                self.remove_token_from_document(token, docid)
            token_inc = self.document_metadata[docid]['length']
            del self.document_metadata[docid]
            self.calculate_stats(-1, -token_inc)
        except:
            raise KeyError(f'{docid} not in index')
    
    def add_doc(self, docid: int, tokens: list[str]) -> None:
        token_counts = Counter(tokens)
        self.document_metadata[docid] = {
            'length': len(tokens), 
            'doc_token_count': len(set(tokens) - set([""])),
            'fake_id': self.add_doc_to_fake(docid),
        }
        for token, count in token_counts.items():
            if token != "":
                self.add_token_to_document(token, docid, count)
        self.calculate_stats(1, len(tokens))
        if self.statistics['number_of_documents'] % 1000 == 0:
            self.index.sync()
    
    def get_postings(self, term: str) -> list:
        return self.index.get(term, [])

    def get_doc_metadata(self, docid: int) -> dict[str, int]:
        return self.document_metadata.get(docid, {})
    
    def get_term_metadata(self, term: str) -> dict[str, int]:
        return {
            "doc_freq": len(self.index.get(term, {})),
            "term_freq": self.statistics['vocab'].get(term, 0),
        } if term in self.vocabulary else None
    
    def get_statistics(self) -> dict[str, int]:
        return self.statistics
    
    def save(self) -> None:
        self.create_dir_if_not_exist()
        self.index.sync()
        self.save_except_index()

    def load(self) -> None:
        self.create_dir_if_not_exist()
        self.index = shelve.open(f'{self.index_name}/{self.index_name}', flag='c', writeback=True)
        self.load_except_index()



class BasicInvertedIndexWithSPIMI(BasicInvertedIndex):
    """BasicInvertedIndex with somehow SPIMI
    
    1. flush the index to disk when the number of documents reaches 10w
    2. load the index from disk when the index is loaded
    3. statistics will be calculated in the load_spimi as the stored values is not the whole index
    4. 2x quicker than BasicInvertedIndex after the calculation of statistics is redesigned
    """
    def __init__(self, index_name) -> None:
        super().__init__(index_name)
        self.statistics['index_type'] = 'BasicInvertedIndexWithSPIMI'

    def remove_doc(self, docid: int) -> None:
        try:
            for token in self.vocabulary.copy():
                self.remove_token_from_document(token, docid)
            token_inc = self.document_metadata[docid]['length']
            del self.document_metadata[docid]
            self.calculate_stats(-1, -token_inc)
        except:
            raise KeyError(f'{docid} not in index')

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        token_counts = Counter(tokens)
        self.document_metadata[docid] = {
            'length': len(tokens), 
            'doc_token_count': len(set(tokens) - set([""])),
            'fake_id': self.add_doc_to_fake(docid),
        }
        for token, count in token_counts.items():
            if token != "":
                self.add_token_to_document(token, docid, count)
        self.calculate_stats(1, len(tokens))
        if self.statistics['number_of_documents'] % 100000 == 0 and self.statistics['number_of_documents'] != 0:
            self.flush_to_disk()
    
    def save(self) -> None:
        self.create_dir_if_not_exist()
        self.flush_to_disk()
        self.save_except_index_metadata()
    
    def load(self) -> None:
        self.create_dir_if_not_exist()
        self.index = self.load_spimi('index')
        self.document_metadata = self.load_spimi('metadata')
        self.load_except_index_metadata()

    def load_spimi(self, match_name: str) -> None:
        obj_spimi = {}
        files = os.listdir(self.index_name)
        filtered_files = list(filter(lambda x: x.startswith(f'{self.index_name}_{match_name}_'), files))
        ordered_files = sorted(filtered_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.index_segment = len(ordered_files)
        for file in tqdm(ordered_files):
            with open(f'{self.index_name}/{file}', 'r') as f:
                infos = json.load(f)
                if match_name == 'metadata':
                    infos = {int(k): v for k, v in infos.items()}
                    obj_spimi.update(infos)
                elif match_name == 'index':
                    for k, v in infos.items():
                        if k not in obj_spimi:
                            obj_spimi[k] = []
                        obj_spimi[k].extend(v)
        return obj_spimi
    
    def flush_to_disk(self) -> None:
        self.create_dir_if_not_exist()
        with open(f'{self.index_name}/{self.index_name}_index_{self.index_segment}.json', 'w') as f:
            json.dump(self.index, f)
        with open(f'{self.index_name}/{self.index_name}_metadata_{self.index_segment}.json', 'w') as f:
            json.dump(self.document_metadata, f)
        self.index_segment += 1
        self.index = {}
        self.document_metadata = {}

    def save_except_index_metadata(self) -> None:
        with open(f'{self.index_name}/{self.index_name}_statistics.json', 'w') as f:
            json.dump(self.statistics, f)
        with open(f'{self.index_name}/{self.index_name}_vocabulary.json', 'w') as f:
            json.dump(list(self.vocabulary), f)
    
    def load_except_index_metadata(self) -> None:
        fake_ids = [v['fake_id'] for v in self.document_metadata.values()]
        self.doc_counter = max(fake_ids) + 1
        with open(f'{self.index_name}/{self.index_name}_statistics.json', 'r') as f:
            self.statistics = json.load(f)
        with open(f'{self.index_name}/{self.index_name}_vocabulary.json', 'r') as f:
            self.vocabulary = set(json.load(f))


class Indexer:
    '''The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''
    @staticmethod
    def create_index(index_name: str, index_type: IndexType, dataset_path: str, document_preprocessor, stopword_filtering: bool, minimum_word_frequency: int) -> InvertedIndex:
        '''
        The Index class' static function which is responsible for creating the indexes already created indexes present on disk.

        Parameters:

        index_name [str]: This is essentially the folder where you would keep all of the generated index files. The generated files may differ from student to student based on their implementation.

        index_type [IndexType]: This parameter tells you which type of index to create - Inverted index or positional index.

        dataset_path [str]: This is the path to your dataset

        document_preprocessor: This is a class which has a 'tokenize' function which would read each document's text and return back a list of valid tokens.

        stopword_filtering [bool]: This is an optional configuration where you could enable or disable stop word filtering.

        minimum_word_frequency [int]: This is also an optional configuration which sets the minimum word frequency of a particular token to be indexed. If the token does not appear in the document atleast for the set frequency, it will not be indexed. Setting a value of 0 will completely ignore the parameter.

        '''
        # TODO implement this class properly. This is responsible for going through the documents one by one and inserting them into the index after tokenizing the document
        # return the index object
        index_type_to_class = {
            IndexType.InvertedIndex: BasicInvertedIndex,
            IndexType.PositionalIndex: PositionalInvertedIndex,
            IndexType.OnDiskInvertedIndex: OnDiskInvertedIndex,
            IndexType.SampleIndex: SampleIndex,
            'BasicInvertedIndexWithSPIMI': BasicInvertedIndexWithSPIMI,
        }
        index = index_type_to_class[index_type](index_name)

        # stopwords filtering
        stopwords = []        
        if stopword_filtering:
            with open('stopwords.txt', 'r') as stopwords_file:
                stopwords = list(map(lambda x: x.strip(), stopwords_file.readlines()))
        stopwords = set(stopwords)
        nonsense_token = ""
        stopwords_mapper = lambda x: nonsense_token if str.lower(x) in stopwords else x

        # loading dataset
        dataset = read_jsonl(dataset_path)

        word_tokenizer = lambda x: str.lower(x)
        for doc in tqdm(dataset, desc='Indexing:'):
            tokens = document_preprocessor.tokenize(doc['text'])
            # stopwords filtering
            if stopword_filtering:
                tokens = list(map(stopwords_mapper, tokens))
            # minimum word frequency
            if minimum_word_frequency > 0:
                mwf_words = []
                lower_word_to_word = defaultdict(list)
                for word in set(tokens):
                    lower_word_to_word[word_tokenizer(word)].append(word)
                lower_token_count = Counter(map(word_tokenizer, tokens))
                for token, count in lower_token_count.items():
                    if count < minimum_word_frequency:
                        mwf_words.extend(lower_word_to_word[token])
                mwf_words = set(mwf_words)
                mwf_words_mapper = lambda x: nonsense_token if x in mwf_words else x
                tokens = list(map(mwf_words_mapper, tokens))
            index.add_doc(doc['docid'], tokens)

        index.save()
        # index.load()
        # load for OnDiskInvertedIndex
        return index


# TODO for each inverted index implementation, use the Indexer to create an index with the first 10, 100, 1000, and 10000 documents in the collection (what was just preprocessed). At each size, record (1) how
# long it took to index that many documents and (2) using the get memory footprint function provided, how much memory the index consumes. Record these sizes and timestamps. Make
# a plot for each, showing the number of documents on the x-axis and either time or memory
# on the y-axis.

'''
The following class is a stub class with none of the essential methods implemented. It is merely here as an example.
'''


class SampleIndex(InvertedIndex):
    '''
    This class does nothing of value
    '''

    def add_doc(self, docid, tokens):
        """Tokenize a document and add term ID """
        for token in tokens:
            if token not in self.index:
                self.index[token] = {docid: 1}
            else:
                self.index[token][docid] = 1
    
    def save(self):
        print('Index saved!')


if __name__ == '__main__':
    import time
    from document_preprocessor import RegexTokenizer, read_jsonl
    from utils import Utils
    from pympler import asizeof

    indexers = [
        # ('BasicInvertedIndex', IndexType("InvertedIndex")),
        # ('PositionalInvertedIndex', IndexType("PositionalIndex")),
        # ('OnDiskInvertedIndex', IndexType("OnDiskInvertedIndex")),
        ('BasicInvertedIndexWithSPIMI', 'BasicInvertedIndexWithSPIMI'),
    ]

    dataset_path = 'wikipedia_1M_dataset.jsonl'
    multi_word_expressions_file = 'multi_word_expressions.txt'
    document_preprocessor = RegexTokenizer(multi_word_expressions_file)

    time_infos = {}
    memory_infos = {}
    for index_name, index_type in indexers:
        start_time = time.time()
        index = Indexer.create_index(index_name, index_type, dataset_path, document_preprocessor, True, 2)
        print("Number of documents, unique tokens, total tokens")
        print(index.statistics["number_of_documents"], index.statistics["unique_token_count"], index.statistics["total_token_count"])
        if index_type == IndexType.OnDiskInvertedIndex:
            index.index.close()
        duration = time.time() - start_time
        # time consuming to calculate the memory footprint
        # memory_footprint = asizeof.asizeof(index)
        print(f'{index_name} indexing time: {duration} seconds')
        # print(f'{index_name} memory footprint: {memory_footprint} bytes')

    # # draw the time and memory performance
    # import seaborn as sns
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # sns.set(style="whitegrid")
    # # draw all the time infos in one plot with different colors

    # time_infos = {
    #     "BasicInvertedIndex": {
    #         10: 0.47903013229370117,
    #         100: 2.8924648761749268,
    #         1000: 14.326866149902344,
    #         10000: 70.54033041000366
    #     },
    #     "PositionalInvertedIndex": {
    #         10:  0.6474723815917969,
    #         100: 4.307590961456299,
    #         1000: 22.547844409942627,
    #         10000: 85.18972682952881
    #     },
    #     "OnDiskInvertedIndex": {
    #         10: 10.817692041397095,
    #         100: 40.68445181846619,
    #         1000: 113.73002934455872,
    #         10000: 496.71371626853943,
    #     },
    # }

    # memory_infos = {
    #     "BasicInvertedIndex": {
    #         10: 3677312,
    #         100: 18964496,
    #         1000: 73546280,
    #         10000:  270940896
    #     },
    #     "PositionalInvertedIndex": {
    #         10: 10094120,
    #         100: 60176504,
    #         1000: 268571152,
    #         10000: 946997896
    #     },
    #     "OnDiskInvertedIndex": {
    #         10: 1279960,
    #         100: 4935856,
    #         1000: 9084640,
    #         10000: 21804616,
    #     },
    # }

    # def draw_func(time_infos, draw_type='time'):
    #     df = pd.DataFrame(time_infos).reset_index()
    #     df = df.rename(columns={'index': 'num_docs'})
    #     df = pd.melt(df, id_vars=['num_docs'], value_vars=['BasicInvertedIndex', 'PositionalInvertedIndex', 'OnDiskInvertedIndex'])
    #     df = df.rename(columns={'variable': 'index_type', 'value': 'time'})
    #     ax = sns.lineplot(x="num_docs", y="time", hue="index_type", data=df)
    #     if draw_type == 'time':
    #         ax.set(xlabel='Number of Documents', ylabel='Time (s)')
    #         ax.set_title('Time Performance of Different Index Types')
    #     elif draw_type == 'memory':
    #         ax.set(xlabel='Number of Documents', ylabel='Memory (bytes)')
    #         ax.set_title('Memory Performance of Different Index Types')
    #     plt.savefig('{}.png'.format(draw_type))
    
    # draw_func(time_infos, draw_type='time')
    # draw_func(memory_infos, draw_type='memory')