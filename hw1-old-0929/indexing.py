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


    # NOTE: The following functions have to be implemented in the three inherited classes and not in this class
    

    def remove_doc(self, docid: int) -> None:
        # TODO implement this to remove a document from the entire index and statistics
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        # TODO implement this to add documents to the index
        raise NotImplementedError

    def get_postings(self, term: str) -> dict[str|int, int|list]:
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
    
    def calculate_stats(self) -> None:
        # TO-DO: not use self.index
        self.statistics['unique_token_count'] = len(self.vocabulary)
        self.statistics['total_token_count'] = sum(self.document_metadata[docid]['length'] for docid in self.document_metadata)
        self.statistics['number_of_documents'] = len(self.document_metadata)
        if self.statistics['number_of_documents'] == 0:
            self.statistics['mean_document_length'] = 0
        else:
            self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents']

    def create_dir_if_not_exist(self) -> None:
        if not os.path.exists(f'{self.index_name}'):
            os.mkdir(f'{self.index_name}')

    def remove_token_from_document(self, token: str, docid: int) -> None:
        if token in self.index and docid in self.index[token]:
            # statistics
            token_count = self.index[token][docid]
            if isinstance(token_count, list):
                token_count = len(token_count)
            self.statistics['vocab'][token] -= token_count
            if self.statistics['vocab'][token] <= 0:
                del self.statistics['vocab'][token]
            # index
            del self.index[token][docid]
            if len(self.index[token]) == 0:
                del self.index[token]
                self.vocabulary.remove(token)
        
    
    def add_token_to_document(self, token: str, docid: int, count: int) -> None:
        # index
        if token not in self.index:
            self.index[token] = {docid: count}
            self.vocabulary.add(token)
        else:
            self.index[token][docid] = count
        # statistics
        if token not in self.statistics['vocab']:
            self.statistics['vocab'][token] = 0
        self.statistics['vocab'][token] += count

    def add_token_to_pos_doc(self, token: str, docid: int, position: int) -> None:
        # index
        if token not in self.index:
            self.index[token] = {docid: [position]}
            self.vocabulary.add(token)
        elif docid not in self.index[token]:
            self.index[token][docid] = [position]
        else:
            self.index[token][docid].append(position)
        # statistics
        if token not in self.statistics['vocab']:
            self.statistics['vocab'][token] = 0
        self.statistics['vocab'][token] += 1

    def save_except_index(self) -> None:
        with open(f'{self.index_name}/{self.index_name}_metadata.json', 'w') as f:
            json.dump(self.document_metadata, f)
        with open(f'{self.index_name}/{self.index_name}_statistics.json', 'w') as f:
            json.dump(self.statistics, f)
        with open(f'{self.index_name}/{self.index_name}_vocabulary.json', 'w') as f:
            json.dump(list(self.vocabulary), f)
    
    def load_except_index(self) -> None:
        with open(f'{self.index_name}/{self.index_name}_metadata.json', 'r') as f:
            self.document_metadata = json.load(f)
            # convert str to int
            self.document_metadata = {int(k): v for k, v in self.document_metadata.items()}
        with open(f'{self.index_name}/{self.index_name}_statistics.json', 'r') as f:
            self.statistics = json.load(f)
        with open(f'{self.index_name}/{self.index_name}_vocabulary.json', 'r') as f:
            self.vocabulary = set(json.load(f))

    # def remove_token_from_index(self, token: str) -> None:
    #     if token in self.index:
    #         # index
    #         for docid, positions in self.index[token].copy().items():
    #             # document_metadata
    #             if isinstance(positions, list):
    #                 positions = len(positions)
    #             # self.document_metadata[docid]['length'] -= positions
    #             # self.document_metadata[docid]['unique_token_count'] -= 1
    #             self.remove_token_from_document(token, docid)
    #         # statistics
    #         # token_count = self.statistics['vocab'][token]
    #         # self.calculate_stats(-token_count)
    #         # vocabulary
    #         if token in self.index:
    #             del self.index[token]
    #             self.vocabulary.remove(token)
    #             del self.statistics['vocab'][token]
    #         self.calculate_stats(0)


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
            del self.document_metadata[docid]
            self.calculate_stats()
        except:
            raise KeyError(f'{docid} not in index')

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        token_counts = Counter(tokens)
        self.document_metadata[docid] = {
            'length': len(tokens), 
            'unique_token_count': len(set(tokens)),
        }
        for token, count in token_counts.items():
            if token != "":
                self.add_token_to_document(token, docid, count)
        self.calculate_stats()
    
    def get_postings(self, term: str) -> dict[str|int, int|list]:
        return self.index.get(term, {})
    
    def get_doc_metadata(self, docid: int) -> dict[str, int]:
        return self.document_metadata.get(docid, {})

    def get_term_metadata(self, term: str) -> dict[str, int]:
        return {"doc_freq": len(self.index.get(term, {}))} if term in self.vocabulary else None
    
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
            # convert str to int
            self.index = {k: {int(kk): vv for kk, vv in v.items()} for k, v in self.index.items()}
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
            del self.document_metadata[docid]
            self.calculate_stats()
        except:
            raise KeyError(f'{docid} not in index')

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        self.document_metadata[docid] = {
            'length': len(tokens), 
            'unique_token_count': len(set(tokens))
        }
        for i, token in enumerate(tokens):
            if token != "":
                self.add_token_to_pos_doc(token, docid, i)
        self.calculate_stats()
    
    def get_postings(self, term: str) -> dict[str|int, int|list]:
        return self.index.get(term, {})
        
    def get_doc_metadata(self, docid: int) -> dict[str, int]:
        return self.document_metadata.get(docid, {})
    
    def get_term_metadata(self, term: str) -> dict[str, int]:
        return {"doc_freq": len(self.index.get(term, {}))} if term in self.vocabulary else None
    
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
            # convert str to int
            self.index = {k: {int(kk): vv for kk, vv in v.items()} for k, v in self.index.items()}
        self.load_except_index()
        

class OnDiskInvertedIndex(InvertedIndex):
    def __init__(self, index_name) -> None:
        super().__init__(index_name)
        self.statistics['index_type'] = 'OnDiskInvertedIndex'
        self.create_dir_if_not_exist()
        self.index = shelve.open(f'{self.index_name}/{self.index_name}', writeback=True)
    # TODO implement all the functions mentioned in the interface
    # This is a typical inverted index which will be using Python's shelve module to persist and even read the data from the disk.

    def remove_doc(self, docid: int) -> None:
        try:
            for token in self.vocabulary.copy():
                self.remove_token_from_document(token, docid)
            del self.document_metadata[docid]
            self.calculate_stats()
        except:
            raise KeyError(f'{docid} not in index')
    
    def add_doc(self, docid: int, tokens: list[str]) -> None:
        token_counts = Counter(tokens)
        self.document_metadata[docid] = {
            'length': len(tokens), 
            # might be wrong
            'unique_token_count': len(set(tokens))
        }
        for token, count in token_counts.items():
            if token != "":
                self.add_token_to_document(token, docid, count)
        self.calculate_stats()
        if self.statistics['number_of_documents'] % 1000 == 0:
            self.index.sync()
    
    def get_postings(self, term: str) -> dict[str | int, int | list]:
        return self.index.get(term, {})

    def get_doc_metadata(self, docid: int) -> dict[str, int]:
        return self.document_metadata.get(docid, {})
    
    def get_term_metadata(self, term: str) -> dict[str, int]:
        return {"doc_freq": len(self.index.get(term, {}))} if term in self.vocabulary else None
    
    def get_statistics(self) -> dict[str, int]:
        return self.statistics
    
    def save(self) -> None:
        self.create_dir_if_not_exist()
        self.index.sync()
        # self.index.close()
        self.save_except_index()

    def load(self) -> None:
        self.create_dir_if_not_exist()
        self.index = shelve.open(f'{self.index_name}/{self.index_name}', writeback=True)
        # convert str to int
        self.index = {k: {int(kk): vv for kk, vv in v.items()} for k, v in self.index.items()}
        self.load_except_index()
    
    def close(self) -> None:
        self.index.close()


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
        # for doc in tqdm(dataset_path):
        dataset = read_jsonl(dataset_path)

        word_tokenizer = lambda x: str.lower(x)
        for doc in tqdm(dataset):
            tokens = document_preprocessor.tokenize(doc['text'])
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
    indexers = [
        ('BasicInvertedIndex', IndexType("InvertedIndex")),
        ('PositionalInvertedIndex', IndexType("PositionalIndex")),
        ('OnDiskInvertedIndex', IndexType("OnDiskInvertedIndex")),
    ]

    dataset_path = 'wikipedia_1M_dataset.jsonl'
    multi_word_expressions_file = 'multi_word_expressions.txt'
    document_preprocessor = RegexTokenizer(multi_word_expressions_file)

    for index_name, index_type in indexers:
        start_time = time.time()
        index = Indexer.create_index(index_name, index_type, dataset_path, document_preprocessor, True, 2)
        duration = time.time() - start_time
        memory_footprint = Utils.get_memory_footprint(index.index)
        print(f'{index_name} indexing time: {duration} seconds')
        print(f'{index_name} memory footprint: {memory_footprint} bytes')