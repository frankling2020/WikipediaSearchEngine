'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
Use libraries such as tqdm, orjson, collections.Counter, shelve if you need them.
DO NOT use the pickle module.
NOTE: 
There are a few changes to the indexing file for HW2.
The changes are marked with a comment `# NOTE: changes in this method for HW2`. 
Please see more in the README.md.
'''
from enum import Enum
import json
import os
from tqdm import tqdm
from collections import Counter, defaultdict
from document_preprocessor import Tokenizer
import gzip

from bisect import bisect_left
import orjson

class IndexType(Enum):
    # the three types of index currently supported are InvertedIndex, PositionalIndex and OnDiskInvertedIndex
    InvertedIndex = 'BasicInvertedIndex'
    # NOTE: You don't need to support these other three
    PositionalIndex = 'PositionalIndex'
    OnDiskInvertedIndex = 'OnDiskInvertedIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    '''
    The base interface representing the data structure for all index classes.
    The functions are meant to be implemented in the actual index classes and not as part of this interface.
    '''

    def __init__(self) -> None:
        self.statistics = defaultdict(Counter)  # the central statistics of the index
        self.index = {}  # the index
        # metadata like length, number of unique tokens of the documents
        self.document_metadata = {}
        self.vocabulary = set()
        self.__init__helper()

    # NOTE: The following functions have to be implemented in the three inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        # TODO implement this to remove a document from the entire index and statistics
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        '''
        Adds a document to the index and updates the index's metadata on the basis of this
        document's addition (e.g., collection size, average document length, etc.)

        Arguments:
            docid [int]: the identifier of the document

            tokens list[str]: the tokens of the document. Tokens that should not be indexed will have 
            been replaced with None in this list. The length of the list should be equal to the number
            of tokens prior to any token removal.
        '''
        # TODO implement this to add documents to the index
        raise NotImplementedError

    def get_postings(self, term: str) -> list[tuple[int, int]]:
        '''
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.
        
        Arguments:
            term [str]: the term to be searched for

        Returns:
            list[tuple[int,str]] : A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in 
            the document.
        '''
        # TODO implement this to fetch a term's postings from the index
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        '''
        For the given document id, returns a dictionary with metadata about that document. Metadata
        should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)             
        '''
        # TODO implement to fetch a particular documents stored metadata
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        '''
        For the given term, returns a dictionary with metadata about that term in the index. Metadata
        should include keys such as the following:
            "count": How many times this term appeared in the corpus as a whole.          
        '''        
        # TODO implement to fetch a particular terms stored metadata
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        '''
        Returns a dictionary mapping statistical properties (named as strings) about the index to their values.  
        Keys should include at least the following:

            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens), 
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)                
        '''
        # TODO calculate statistics like 'unique_token_count', 'total_token_count', 
        #  'number_of_documents', 'mean_document_length' and any other relevant central statistic.
        raise NotImplementedError

    # NOTE: changes in this method for HW2
    def save(self, index_directory_name) -> None:
        '''
        Saves the state of this index to the provided directory. The save state should include the
        inverted index as well as any meta data need to load this index back from disk
        '''
        # TODO save the index files to disk
        raise NotImplementedError

    # NOTE: changes in this method for HW2
    def load(self, index_directory_name) -> None:
        '''
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save()
        '''
        # TODO load the index files from disk to a Python object
        raise NotImplementedError
    
    ###############################################################################################
    #### SELF-DEFINED FUNCTIONS                                                                ####
    ###############################################################################################
    def __init__helper(self) -> None:
        self.statistics = {
            'mean_document_length': 0, 
            'number_of_documents': 0, 
            'total_token_count': 0, 
            'unique_token_count': 0, 
            'total_token_count': 0,
            'fake_id_counter': 0,
        }
        self.term_freq = defaultdict(int)

    def save_json_func(self, data, f):
        if isinstance(data, dict):
            data = {str(k): v for k, v in data.items()}
        f.write(orjson.dumps(data).decode('utf-8'))
        # json.dump(data, f)

    def load_json_func(self, f):
        return orjson.loads(f.read())
        # return json.load(f)

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

    def create_dir_if_not_exist(self, index_directory_name) -> None:
        if not os.path.exists(f'{index_directory_name}'):
            os.mkdir(f'{index_directory_name}')
    
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
        if on_disk:
            tmp = self.index[token]
            tmp.append([docid, count])
            self.index[token] = tmp
        else:
            self.index[token].append((docid, count))

    def save_except_index(self, index_directory_name) -> None:
        # in practice spimi will treat metadata
        index_type = self.statistics['index_type']
        with open(f'{index_directory_name}/{index_type}_metadata.json', 'w') as f:
            self.save_json_func(self.document_metadata, f)
        with open(f'{index_directory_name}/{index_type}_statistics.json', 'w') as f:
            self.save_json_func(self.statistics, f)
        with open(f'{index_directory_name}/{index_type}_vocabulary.json', 'w') as f:
            self.save_json_func(list(self.vocabulary), f)            

    def load_except_index(self, index_directory_name) -> None:
        # spimi will treat metadata
        index_type = self.statistics['index_type']
        with open(f'{index_directory_name}/{index_type}_metadata.json', 'r') as f:
            self.document_metadata = self.load_json_func(f)
            # convert str to int
            self.document_metadata = {int(k): v for k, v in self.document_metadata.items()}
        with open(f'{index_directory_name}/{index_type}_statistics.json', 'r') as f:
            self.statistics = self.load_json_func(f)
        with open(f'{index_directory_name}/{index_type}_vocabulary.json', 'r') as f:
            self.vocabulary = set(self.load_json_func(f))
    
    def add_doc_to_fake(self, docid: int) -> int:
        fake_id = self.statistics['fake_id_counter']
        self.statistics['fake_id_counter'] = fake_id + 1
        return fake_id


class BasicInvertedIndex(InvertedIndex):
    '''
    An inverted index implementation where everything is kept in memory.
    '''

    # NOTE: changes in this class for HW2
    def __init__(self) -> None:
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
    # for example, you can initialize the index and statistics here:
    #    self.statistics['docmap'] = {}
    #    self.index = defaultdict(list)
    #    self.doc_id = 0

    # TODO implement all the functions mentioned in the InvertedIndex base class
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
        del token_counts
        self.calculate_stats(1, len(tokens))
    
    def get_postings(self, term: str) -> list[tuple[int, int]]:
        return self.index.get(term, [])
    
    def get_doc_metadata(self, docid: int) -> dict[str, int]:
        return self.document_metadata.get(docid, {})

    def get_term_metadata(self, term: str) -> dict[str, int]:
        term_freq = self.term_freq.get(term, 0)
        if term_freq == 0 and term in self.vocabulary:
            for x in self.index[term]:
                term_freq += x[1]
            self.term_freq[term] = term_freq
        return {
            "doc_freq": len(self.index.get(term, {})),
            "term_freq": self.term_freq.get(term, 0),
        } if term in self.vocabulary else None
    
    def get_statistics(self) -> dict[str, int]:
        return self.statistics
    
    def save(self, index_directory_name) -> None:
        self.create_dir_if_not_exist(index_directory_name)
        index_type = self.statistics['index_type']
        with open(f'{index_directory_name}/{index_type}_index.json', 'w') as f:
            self.save_json_func(self.index, f)
        self.save_except_index(index_directory_name)

    def load(self, index_directory_name) -> None:
        self.create_dir_if_not_exist(index_directory_name)
        index_type = self.statistics['index_type']
        with open(f'{index_directory_name}/{index_type}_index.json', 'r') as f:
            self.index = self.load_json_func(f)
            for token in tqdm(self.index, desc="load index"):
                self.index[token] = list(map(lambda x: tuple(x), self.index[token]))
                for term_info in self.index[token]:
                    token = term_info[0]
                    count = term_info[1]
                    self.term_freq[token] += count
        self.load_except_index(index_directory_name)


class PositionalInvertedIndex(BasicInvertedIndex):
    '''
     This is the positional inverted index where each term keeps track of documents and positions of the terms occring in the document.
    '''
    def __init__(self, index_name) -> None:
        super().__init__(index_name)
        self.statistics['index_type'] = 'PositionalInvertedIndex'
        # for example, you can initialize the index and statistics here:
        # self.statistics['offset'] = [0]
        # self.statistics['docmap'] = {}
        # self.doc_id = 0
        # self.postings_id = -1

    # TODO: Do nothing, unless you want to explore using a positional index for some cool features


class OnDiskInvertedIndex(BasicInvertedIndex):
    '''
    This is an inverted index where the inverted index's keys (words) are kept in memory but the
    postings (list of documents) are on desk. The on-disk part is expected to be handled via
    a library.
    '''
    def __init__(self, shelve_filename) -> None:
        super().__init__()
        self.shelve_filename = shelve_filename
        self.statistics['index_type'] = 'OnDiskInvertedIndex'
        # # Ensure that the directory exists        
        # self.index = shelve.open(self.shelve_filename, 'index')
        # self.statistics['docmap'] = {}
        # self.doc_id = 0

    # NOTE: Do nothing, unless you want to re-experience the pain of cross-platform compatibility :'( 



def read_dataset(dataset_path: str, max_docs: int = -1):
    """Read the dataset from the path with a maximum number of documents to read

    Args:
        dataset_path (str): dataset path
        max_docs (int, optional): maximum number of documents to read. Defaults to -1.

    Yields:
        dict: a document
    """
    open_func = lambda x: gzip.open(x, 'rb') if x.endswith('.gz') else open(x, 'r')
    with open_func(dataset_path) as f:
        for i, line in enumerate(f):
            if max_docs != -1 and i >= max_docs:
                break
            yield json.loads(line)


class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''

    # NOTE: changes in this method for HW2. 
    # See more in the README.md.
    # replaced argument: `stopword_filtering`
    # new argument: `text_key` & `max_docs`
    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str, 
                     document_preprocessor: Tokenizer, stopwords: set[str], 
                     minimum_word_frequency: int, text_key="text",
                     max_docs=-1) -> InvertedIndex:
        '''
        The Index class' static function which is responsible for creating an inverted index

        Parameters:        

        index_type [IndexType]: This parameter tells you which type of index to create, e.g., a BasicInvertedIndex.

        dataset_path [str]: This is the file path to your dataset

        document_preprocessor: This is a class which has a 'tokenize' function which would read each document's text and return back a list of valid tokens.

        stopwords [set[str]]: The set of stopwords to remove during preprocessing or `None` if no stopword preprocessing is to be done.

        minimum_word_frequency [int]: This is also an optional configuration which sets the minimum word frequency of a particular token to be indexed. If the token does not appear in the document atleast for the set frequency, it will not be indexed. Setting a value of 0 will completely ignore the parameter.

        text_key [str]: the key in the JSON to use for loading the text. 

        max_docs [int]: The maximum number of documents to index. Documents are processed in the order they are seen

        '''        


        # TODO: Figure out what type of InvertedIndex to create. For homework 2, only the BasicInvertedIndex is required
        # to be supported
        index_type_to_class = {
            IndexType.InvertedIndex: BasicInvertedIndex,
            IndexType.PositionalIndex: PositionalInvertedIndex,
            IndexType.OnDiskInvertedIndex: OnDiskInvertedIndex,
        }

        index = index_type_to_class[index_type]()            

        # TODO: If minimum word frequencies are specified, process the collection to get the
        # word frequencies
        # NOTE (hw2): `minimum_word_frequency` needs to be computed based on frequencies 
        # at the **collection-level***, _not_ at the document level.

        nonsense_token = ""
        stopwords = set(map(str.lower, stopwords))
        stopwords_mapper = lambda x: nonsense_token if str.lower(x) in stopwords else x

        # NOTE: Make sure to support both .jsonl.gz and .jsonl as input.

        # TODO: Figure out which set of words to not index because they are stopwords or
        # have too low of a frequency.
        # minimum frequency word
        mwf_words = set()
        if minimum_word_frequency > 0:
            token_counter = defaultdict(int)
            for doc in tqdm(read_dataset(dataset_path, max_docs), desc="min_freq"):
                tokens = document_preprocessor.tokenize(doc[text_key])
                for token in tokens:
                    count = token_counter[token]
                    if count < minimum_word_frequency:
                        token_counter[token] = count + 1 
            mwf_words = set(filter(lambda x: token_counter[x] < minimum_word_frequency, token_counter.keys()))
        mwf_words_mapper = lambda x: nonsense_token if x in mwf_words else x
        
        for doc in tqdm(read_dataset(dataset_path, max_docs), desc="indexing"):
            tokens = document_preprocessor.tokenize(doc[text_key])
            for i, token in enumerate(tokens):
                token = stopwords_mapper(token)
                if token != nonsense_token:
                    token = mwf_words_mapper(token)
                tokens[i] = token
            docid = doc['docid']
            index.add_doc(docid, tokens)


        # HINT: This homework should work fine on a laptop with 8GB of memory but if you need, you can
        # delete some unused objects to free up some space

        # TODO: Once the set of stopwords and low-frequency words is determined, read the collection 
        # and process/index each document. Only index the terms that are not stopwords and have 
        # high-enough frequency. This should involve replacing those tokens with None values and 
        # passing the documents to your inverted index object's add_doc

        index.save('BasicInvertedIndex')
        return index
    
    @staticmethod
    def load_index(index_directory_name: str) -> InvertedIndex:
        '''
        Loads an inverted index from the specified directory
        '''
        index = BasicInvertedIndex()
        index.load(index_directory_name)
        return index


###############################################################################################
#### P1 Important categories                                                               ####
###############################################################################################
def get_important_categories(dataset_path: str, minimum_category_count: int = 1000) -> None:
    """Get the set of categories that occur more than minimum_category_count times

    Args:
        dataset_path (str): dataset path
    """
    important_categories = defaultdict(int)
    for doc in tqdm(read_dataset(dataset_path)):
        for category in doc['categories']:
            if important_categories[category] < minimum_category_count:
                important_categories[category] += 1
    important_categories = set([k for k, v in important_categories.items() if v >= minimum_category_count])
    # save the important categories to a file
    with open('important_categories.txt', 'w') as f:
        f.write('\n'.join(important_categories))


if __name__ == '__main__':
    import time
    from document_preprocessor import RegexTokenizer

    indexers = [
        ('BasicInvertedIndex', IndexType.InvertedIndex),
    ]

    # dataset_path = 'wikipedia_200k_dataset.jsonl.gz'
    dataset_path = 'toy_dataset.jsonl'
    document_preprocessor = RegexTokenizer(token_regex='\\w+')

    # p1: find the important categories
    # get_important_categories(dataset_path)

    time_infos = {}
    memory_infos = {}
    stopwords = set()
    with open('stopwords.txt', 'r') as f:
        stopwords = set(map(lambda x: x.strip(), f.readlines()))
    for index_name, index_type in indexers:
        start_time = time.time()
        index = Indexer.create_index(index_type, dataset_path, document_preprocessor, stopwords, 50)
        # index = Indexer.load_index(index_name)
        if index_type == IndexType.OnDiskInvertedIndex:
            index.index.close()
        duration = time.time() - start_time
        print(f'{index_name} took {duration} seconds')