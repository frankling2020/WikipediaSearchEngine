import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pipeline import SearchEngine
from ranker import WordCountCosineSimilarity, TF_IDF, BM25, DirichletLM, YourRanker
from sample_data import SAMPLE_DOCS
from indexing import IndexType


def map_score(actual, ideal, cut_off=10):
    # TODO Implement MAP score metric
    # Here you calculate the Mean Average Precision score for each query

    # Note: actual and ideal are lists of lists. Each list is a query. Each element in the list is a document id.
    # Note: The length of each list is the number of documents retrieved for a query.
    # Note: The length of actual and ideal is the number of queries.
    # Note: The document ids are the rank of the document.
    ideal = set(ideal)
    correct_pos = [pos for pos in range(cut_off) if actual[pos] in ideal]
    precision_k = [(i + 1) / (pos + 1) for i, pos in enumerate(correct_pos)]
    return np.mean(precision_k) if len(precision_k) > 0 else 0


def ndcg_score(actual, ideal, cut_off=10):
    # TODO Implement NDCG score metric
    # Here you calculate the Normalized Discounted Cumulative Gain score for each query
    correct_pos = [pos for pos in range(cut_off) if actual[pos] in ideal]
    for pos in range(min(len(actual), cut_off)):
        if actual[pos] in ideal:
            correct_pos.append((pos, 2 if actual[pos] == ideal[0] else 1))
    dcg = np.sum([gain / np.log2(pos + 2) for pos, gain in correct_pos]) if len(correct_pos) > 0 else 0
    ideal_len = min(len(ideal), cut_off)
    idcg = np.sum([1 / np.log2(pos + 2) for pos in range(ideal_len)]) + 1 if ideal_len > 0 else 0
    return dcg / idcg if idcg > 0 else 0


def run_relevance_tests(algorithm):
    # TODO Implement running relevance test for the whole search system for multiple queries
    '''
    This function is responsible for measuring the the performance of the whole system using metrics such as MAP and NDCG.
    
    Parameters:
    
    algorithm: This is the overall algorithm used by the system to search through the document collection.
    '''
    # 1. Load the relevance dataset.
    relevance_df = pd.read_csv('relevance.csv')
    # 2. Run all of the dataset queries on the search algorithm.
    queries = relevance_df['query'].unique()
    query_rel = relevance_df.groupby('query').apply(lambda x: list(x.sort_values('qid')['docid'])).to_dict()
    
    map_scores = []
    ndcg_scores = []
    
    for query in queries:
        relevance_doc = query_rel[query]
        results = algorithm.search(query)
        result_doc = [result.docid for result in results]
        map_scores.append(map_score(result_doc, relevance_doc))
        ndcg_scores.append(ndcg_score(result_doc, relevance_doc))
    # 3. Get the MAP and NDCG for every single query and average them out.
    map_avg_score = np.mean(map_scores)
    ndcg_avg_score = np.mean(ndcg_scores)
    # 4. Return the scores to the calling function.
    return {'map': map_avg_score, 'ndcg': ndcg_avg_score}

# TODO Score each of the ranking functions on the data we provide. Use the default
# hyperparameters in the code. Plot these scores on the y-axis and relevance function on the
# x-axis using a bar plot. Use different hues for each metric.

def test_all_rank_func():
    # index_name = 'BasicInvertedIndex'
    index_name = 'OnDiskInvertedIndex'
    all_eval_res = []
    for scorer_class in [WordCountCosineSimilarity, TF_IDF, BM25, DirichletLM, YourRanker]:
        search_obj = SearchEngine(
            index_name, 
            dataset_path="wikipedia_1M_dataset.jsonl", 
            mwe_file_path='multi_word_expressions.txt',
            index_class=IndexType.OnDiskInvertedIndex,
            scorer_class=scorer_class,
        )
        eval_res = run_relevance_tests(search_obj)
        all_eval_res.append(eval_res)
        print(f'{scorer_class.__name__} MAP: {eval_res["map"]}, NDCG: {eval_res["ndcg"]}')
    # plot all the scores
    sns.lineplot(data=pd.DataFrame(all_eval_res), markers=True)
    plt.savefig('relevance.png')


# TODO For the best-performing relevance function, calculate
# the MAP and NDCG scores for each query individually and plot these. Use the default
# hyperparameters in the code. Plot these scores on the y-axis and query number on the
# x-axis using a bar plot (i.e., one bar per query).

if __name__ == '__main__':
    # NOTE: You can use this file on your command line by running 'python relevance.py'
    test_all_rank_func()