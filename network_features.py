import numpy as np
from sknetwork.ranking import PageRank, HITS
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from sknetwork.data import from_edge_list
import gzip


class NetworkFeatures:
    """
    A class to help generate network features such as PageRank scores, HITS hub score and HITS authority scores.
    This class uses the scikit-network library https://scikit-network.readthedocs.io to calculate node ranking values.

    OPTIONAL reads
    1. PageRank: https://towardsdatascience.com/pagerank-algorithm-fully-explained-dc794184b4af
    2. HITS: https://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture4/lecture4.html 
    """
    def load_network(self, network_filename: str, total_edges: int):
        """
        Loads the network from the specified file and returns the network. A network file 
        can be listed using a .csv or a .csv.gz file.

        Args:
            network_filename: The name of a .csv or .csv.gz file containing an edge list
            total_edges: The total number of edges in an edge list

        Returns:
            The loaded network from sknetwork
        """
        # TODO load the network edgelist dataset and return the scikit-network graph

        # NOTE: there are 92650947 edges in the big network we give you. However,
        # do not hard code this value here, as it will cause the auto-grader tests
        # to break

        # NOTE: Trying to create the network from a pandas dataframe will not work 
        # (too much memory). You'll need to read the documentation to figure out how to 
        # load in the network in the most memory-efficient way possible. This is the 
        # "hard part" of this class's implementation as it requires you to think about
        # memory and data representations. 


        # NOTE: your code should support reading both gzip and non-gzip formats


        # NOTE: On a reference laptop, loading the network file's data took ~90 seconds
        # and constructing the network took ~75 seconds. We estimate that the entire 
        # network construction memory requirement is under 5GB based on tests with 
        # the reference implementation.
        open_func = lambda x: gzip.open(x, 'rt') if x.endswith('.gz') else open(x, 'r')
        edge_lists = []
        with open_func(network_filename) as f:
            f.readline()
            for line in tqdm(f, total=total_edges):
                edge = list(map(lambda x: int(x.strip()), line.strip().split(',')))
                edge_lists.append(edge)
        graph = from_edge_list(edge_lists, directed=True)
        return graph

        
    def calculate_page_rank(self, graph, damping_factor=0.85, iterations=100) -> list[float]:
        """
        Calculates the PageRank scores for the provided network and
        returns the PageRank values for all nodes.

        Args:
            graph: A graph from sknetwork
            damping_factor: The complement of the teleport probability for the random walker
                For example, a damping factor of .8 has a .2 probability of jumping after each step.
            iterations: The maximum number of iterations to run when computing PageRank

        Returns:
            The PageRank scores for all nodes in the network (array-like)
        """
        # TODO Use scikit-network to run Pagerank and return Pagerank scores
        pagerank = PageRank(damping_factor=damping_factor, n_iter=iterations)
        scores = pagerank.fit_transform(graph.adjacency)
        return scores

    def calculate_hits(self, graph) -> tuple[list[float], list[float]]:
        """
        Calculates the hub scores and authority scores using the HITS algorithm
        for the provided network and returns the two lists of scores as a tuple.

        Args:
            graph: A graph from sknetwork

        Returns:
            The hub scores and authority scores (in that order) for all nodes in the network
        """
        # TODO Use scikit-network to run HITS and return HITS hub scores and authority scores
        hits = HITS()
        hits.fit(graph.adjacency)
        hub_scores = hits.scores_row_
        authority_scores = hits.scores_col_

        # NOTE When returning the HITS scores, the returned tuple should have the hub scores in index 0 and authority score in index 1
        return (hub_scores, authority_scores)

    def get_all_network_statistics(self, graph) -> DataFrame:
        """
        Calculates the PageRank and the hub scores and authority scores using the HITS algorithm
        for the provided network and returns a pandas DataFrame with columns: 
        'docid', 'pagerank', 'authority_score', and 'hub_score' containing the relevant values
        for all nodes in the network.

        Args:
            graph: A graph from sknetwork

        Returns:
            A pandas DataFrame with columns 'docid', 'pagerank', 'authority_score', and 'hub_score'
            containing the relevant values for all nodes in the network
        """

        # TODO Calculate all the Pagerank and HITS scores for the network graph and store it in a dataframe
        pagerank_scores = self.calculate_page_rank(graph)
        hits_scores = self.calculate_hits(graph)
        graph_df = DataFrame(columns=['docid', 'pagerank', 'authority_score', 'hub_score'])
        # graph_df['docid'] = graph.adjacency.indices
        graph_df['docid'] = graph.names
        graph_df['pagerank'] = pagerank_scores
        graph_df['authority_score'] = hits_scores[1]
        graph_df['hub_score'] = hits_scores[0]
        # NOTE: We use a DataFrame here for efficient storage of the values on disk.
        # However, when you actually use these values, you'll convert this DataFrame
        # to another dictionary-based representation for faster lookup when making
        # the L2R features.

        # NOTE Return the dataframe and save the dataframe as a CSV or JSON
        return graph_df


# Example main function
if __name__ == '__main__':
    from time import time
    nf = NetworkFeatures()
    t1 = time()
    g = nf.load_network('edgelist.csv.gz', 92650947)
    t2 = time()
    print("Loading network took", t2 - t1, "seconds")
    final_df = nf.get_all_network_statistics(g)
    t3 = time()
    print("Calculating network statistics took", t3 - t2, "seconds")
    final_df.to_csv('network_stats.csv')
    t4 = time()
    print("Saving network statistics took", t4 - t3, "seconds")
