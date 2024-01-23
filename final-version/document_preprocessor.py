"""
This is the template for implementing the tokenizer for your search engine.
You will be testing some tokenization techniques and build your own tokenizer.
"""
from nltk.tokenize import RegexpTokenizer
import spacy
import re

class Tokenizer:
    def __init__(self, file_path: str) -> None:
        # TODO: Open the file that contains multi-word expressions and initialize a list of multi-word expressions.
        self.multi_word_expressions = []
        # read the file with multi-word expressions
        with open(file_path, 'r') as f:
            for line in f.readlines():
                self.multi_word_expressions.append(line.strip())
        # sort the multi-word expressions by length
        self.multi_word_expressions.sort(key=len, reverse=True)
        # create a pattern for splitting the text
        # pattern = "|".join(map(re.escape, self.multi_word_expressions))
        # self.pattern = re.compile(f'({pattern})')
        # self.mwe_dict = {expression: self.mwe_tokenize(expression) 
        #     for expression in self.multi_word_expressions}
        # self.inverse_mwe_dict = {token: expression 
        #     for expression, token in self.mwe_dict.items()}
        
        self.post_mwe_dict = {expression: self.tokenize_func(expression) for expression in self.multi_word_expressions}
        
        self.inverse_post_mwe_dict = {}
        for expression, tokens in self.post_mwe_dict.items():
            if tokens[0] not in self.inverse_post_mwe_dict:
                self.inverse_post_mwe_dict[tokens[0]] = []
            self.inverse_post_mwe_dict[tokens[0]].append((len(tokens), tokens, expression))
        
    def entity_merger(self, tokens: list[str]) -> list[str]:
        merged_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in self.inverse_post_mwe_dict:
                isMatch = False
                for length, post_mwe_tokens, expression in sorted(self.inverse_post_mwe_dict[token], reverse=True):
                    if i + length <= len(tokens) and \
                            (tokens[i+length-1] == post_mwe_tokens[-1] or tokens[i+length-1][:-1] == post_mwe_tokens[-1]) \
                            and tokens[i:i+length-1] == post_mwe_tokens[:-1]:
                        merged_tokens.append(expression + tokens[i+length-1][len(post_mwe_tokens[-1]):])
                        i += length - 1
                        isMatch = True
                        break
                if not isMatch:
                    merged_tokens.append(token)
            else:
                merged_tokens.append(token)
            i += 1
        return merged_tokens

    def david_tokenizer(self, text: str) -> list[str]:
        words = self.tokenize_func(text)
        words = self.entity_merger(words)
        return words

    def tokenize_func(self, text: str) -> list[str]:
        pass

    # def phrase_tokenize(self, phrases: list[str]) -> list[str]:
    #     # print(phrases)
    #     multi_word_expressions = set(self.multi_word_expressions)
    #     is_multi_word_expression = lambda x: x in multi_word_expressions
    #     phrases_mark = list(map(is_multi_word_expression, phrases))
    #     phrase_tokenizer = lambda x: [x[0]] if x[1] else self.tokenize_func(x[0].strip())
    #     token_list = list(map(phrase_tokenizer, zip(phrases, phrases_mark)))
    #     tokens = list(chain.from_iterable(token_list))
    #     return tokens

    # def remove_nonsense(self, token: list[str]) -> list[str]:
    #     remove_func = lambda x: x.strip(string.punctuation + string.whitespace)
    #     filtered_token = list(filter(lambda x: x != '', map(remove_func, token)))
    #     return filtered_token
    
    # def __old_pre_tokenize(self, text: str) -> list[str]:
    #     phrases = self.pattern.split(text)
    #     words = self.phrase_tokenize(phrases)
    #     words = self.remove_nonsense(words)
    #     return words
    
    # def mwe_tokenize(self, text: str) -> str:
    #     text = ''.join(re.sub(r'[\s\W]', '', text))
    #     return text

    # def replace_mwe(self, text: str) -> str:
    #     # replace multi-word expressions with the same without spaces
    #     replace_func = lambda match: self.mwe_dict[match.group(0)]
    #     return re.sub(self.pattern, replace_func, text)
    
    # def inverse_mwe(self, phrases: list[str]) -> list[str]:
    #     # replace multi-word expressions without spaces with the same with spaces
    #     for i, phrase in enumerate(phrases):
    #         phraseN = self.mwe_tokenize(phrase)
    #         if phraseN in self.inverse_mwe_dict:
    #             phrases[i] = phrases[i].replace(phraseN, self.inverse_mwe_dict[phraseN])
    #     return phrases
    
    # def pre_tokenizer(self, text: str) -> list[str]:
    #     text = self.replace_mwe(text)
    #     phrases = self.tokenize_func(text)
    #     phrases = self.inverse_mwe(phrases)
    #     return phrases


class SampleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        """This is a dummy tokenizer.

        Parameters:

        text [str]: This is an input text you want to tokenize.
        """
        return ['token_1', 'token_2', 'token_3']  # This is not correct; it is just a placeholder.


class SplitTokenizer(Tokenizer):
    def tokenize_func(self, text: str) -> list[str]:
        return text.split()

    def tokenize(self, text: str) -> list[str]:
        # TODO: Implement a tokenizer that uses the split function.
        """Split a string into a list of tokens using whitespace as a delimiter.

        Parameters:

        text [str]: This is an input text you want to tokenize.
        """
        # return self.pre_tokenizer(text)
        return self.david_tokenizer(text)


class RegexTokenizer(Tokenizer):
    def tokenize_func(self, text: str) -> list[str]:
        tokenizer = RegexpTokenizer(r'\w+')
        return tokenizer.tokenize(text)

    def tokenize(self, text: str) -> list[str]:
        # TODO: Implement a tokenizer that uses NLTK’s RegexTokenizer
        """Use NLTK’s RegexTokenizer and regular expression patterns to tokenize a string.

        Parameters:

        text [str]: This is an input text you want to tokenize.
        """
        # return self.pre_tokenizer(text)
        return self.david_tokenizer(text)


class SpaCyTokenizer(Tokenizer):
    def __init__(self, file_path: str) -> None:
        self.nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'lemmatizer', 'textcat'])
        self.nlp.add_pipe("merge_entities")
        super().__init__(file_path)

    def tokenize_func(self, text: str) -> list[str]:
        # Use spaCy to tokenize a string
        doc = self.nlp(text)
        words = [token.text for token in doc]
        return words

    def tokenize(self, text: str) -> list[str]:
        # TODO: Implement a tokenizer that uses spaCy to process named entities as single words
        """Use a spaCy tokenizer to convert named entities into single words.
        Check the spaCy documentation to learn about the feature that supports named entity recognition.

        Parameters:

        text [str]: This is an input text you want to tokenize.
        """
        # return self.pre_tokenizer(text)
        return self.david_tokenizer(text)


def read_jsonl(file_path: str, nlines: int = 0) -> list[dict]:
    """Read the first n lines of a JSONL file.

    Parameters:

    file_path [str]: This is the path to the JSONL file.
    nlines [int]: This is the number of lines to read.
    """
    import json
    from tqdm import trange
    documents = []
    with open(file_path, 'r') as f:
        # number of lines in the file
        if nlines == 0:
            nlines = len(f.readlines())
            f.seek(0)
        for _ in trange(nlines):
            line = f.readline()
            # how to judge to end of file
            if len(line) == 0:
                break
            documents.append(json.loads(line))
    return documents


# TODO tokenize the first 1000 documents and record the time. Make a plot showing the time taken for each.
if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    from tqdm import tqdm, trange
    documents = read_jsonl('wikipedia_1M_dataset.jsonl', 1000)
    times = []

    multi_word_expressions_file = 'multi_word_expressions.txt'
    split_tokenizer = SplitTokenizer(multi_word_expressions_file)
    regex_tokenizer = RegexTokenizer(multi_word_expressions_file)
    spacy_tokenizer = SpaCyTokenizer(multi_word_expressions_file)

    # text = "The United Nations Development Programme is a United Nations agency. The University of California, Los Angeles."

    for tokenizer in [split_tokenizer, regex_tokenizer, spacy_tokenizer]:
        start = time.time()
        for doc in tqdm(documents):
            tokenizer.tokenize(doc['text'])
        end = time.time()
        times.append((end - start) / len(documents))

    plt.bar(['SplitTokenizer', 'RegexTokenizer', 'SpacyTokenizer'], times)
    plt.title('Time taken for tokenization')
    plt.xlabel('Tokenizer')
    plt.ylabel('Time (s)')
    plt.yscale('log')
    plt.savefig('time_taken.png')