from nltk.tokenize import RegexpTokenizer


class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW2.
        """
        # TODO: Save arguments that are needed as fields of this class.
        self.lowercase = lowercase
        self.multiword_expressions = multiword_expressions

    def find_and_replace_mwes(self, input_tokens: list[str]) -> list[str]:
        """
        IGNORE THIS PART; NO NEED TO IMPLEMENT THIS SINCE NO MULTI-WORD EXPRESSION PROCESSING IS TO BE USED.
        For the given sequence of tokens, finds any recognized multi-word expressions in the sequence
        and replaces that subsequence with a single token containing the multi-word expression.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens containing processed multi-word expressions
        """
        # NOTE: You shouldn't implement this in homework 2
        raise NotImplemented("MWE is not supported")
    
    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and returns the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition
        """
        # TODO: Add support for lower-casing.
        if self.lowercase:
            input_tokens = [token.lower() for token in input_tokens]
        return input_tokens
    
    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        raise NotImplementedError('tokenize() is not implemented in the base class; please use a subclass')


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses NLTK's RegexpTokenizer to tokenize a given string.

        Args:
            token_regex: Use the following default regular expression pattern: '\\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW2; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        # TODO: Save a new argument that is needed as a field of this class.
        # TODO: Initialize the NLTK's RegexpTokenizer 
        self.token_regex = token_regex
        self.tokenizer = RegexpTokenizer(self.token_regex)

    def tokenize(self, text: str) -> list[str]:        
        # TODO: Tokenize the given text and perform postprocessing on the list of tokens using the postprocess function.
        tokens = self.tokenizer.tokenize(text)
        tokens = self.postprocess(tokens)
        return tokens

