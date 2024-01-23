import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pipeline import SearchEngine
from ranker import WordCountCosineSimilarity, TF_IDF, BM25, DirichletLM, YourRanker, PivotedNormalization
from sample_data import SAMPLE_DOCS
from indexing import IndexType
from tqdm import tqdm


def map_score(actual, cut_off=10):
    # TODO Implement MAP score metric
    # Here you calculate the Mean Average Precision score for each query

    # Note: actual and ideal are lists of lists. Each list is a query. Each element in the list is a document id.
    # Note: The length of each list is the number of documents retrieved for a query.
    # Note: The length of actual and ideal is the number of queries.
    # Note: The document ids are the rank of the document.
    cut_off = min(len(actual), cut_off)
    all_rel_doc = np.sum([1 if rel > 0 else 0 for rel in actual])
    correct_pos = [pos for pos in range(cut_off) if actual[pos] > 0]
    precision_k = [(i + 1) / (pos + 1) for i, pos in enumerate(correct_pos)]
    return np.sum(precision_k) / all_rel_doc if len(precision_k) > 0 else 0


def ndcg_score(actual, ideal, cut_off=10):
    # TODO Implement NDCG score metric
    # Here you calculate the Normalized Discounted Cumulative Gain score for each query
    actual_len = min(len(actual), cut_off)
    dcg = actual[0] + np.sum([gain / np.log2(pos + 2) for pos, gain in enumerate(actual[1:actual_len])]) if actual_len != 0 else 0
    ideal_len = min(len(ideal), cut_off)
    idcg = ideal[0] +np.sum([gain / np.log2(pos + 2) for pos, gain in enumerate(ideal[1:ideal_len])]) if ideal_len != 0 else 0
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
    unique_query_df = relevance_df[['query', 'qid']].drop_duplicates()
    qid_to_query = {}
    for _, row in unique_query_df.iterrows():
        qid_to_query[row['qid']] = row['query']
    
    map_scores = []
    ndcg_scores = []
    
    for qid, query in tqdm(qid_to_query.items()):
        relevance_doc = relevance_df[relevance_df['qid'] == qid]
        doc_to_rel = {}
        for _, row in relevance_doc.iterrows():
            doc_to_rel[row['docid']] = row['rel']
        results = algorithm.search(query)
        result_rel = [doc_to_rel.get(result.docid, 0) for result in results]
        ideal_rel = sorted(doc_to_rel.values(), reverse=True)
        map_result_rel = list(map(lambda x: 1 if x > 0 else 0, result_rel))
        map_scores.append(map_score(map_result_rel))
        ndcg_scores.append(ndcg_score(result_rel, ideal_rel))
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
    index_name = 'BasicInvertedIndexWithSPIMI'
    # index_name = 'OnDiskInvertedIndex'
    index_type = 'BasicInvertedIndexWithSPIMI'
    all_eval_res = {}
    for scorer_class in [WordCountCosineSimilarity, TF_IDF, BM25, DirichletLM, PivotedNormalization, YourRanker]:
        search_obj = SearchEngine(
            index_name, 
            dataset_path="wikipedia_1M_dataset.jsonl", 
            mwe_file_path='multi_word_expressions.txt',
            index_type=index_type,
            scorer_class=scorer_class,
        )
        eval_res = run_relevance_tests(search_obj)
        all_eval_res[scorer_class.__name__] = eval_res
        print(f'{scorer_class.__name__} MAP: {eval_res["map"]}, NDCG: {eval_res["ndcg"]}')
        if index_name == 'OnDiskInvertedIndex':
            search_obj.index.index.close()
    print(all_eval_res)
    all_eval_res = pd.DataFrame(all_eval_res)
    all_eval_res = all_eval_res.reset_index()
    all_eval_res = all_eval_res.rename(columns={'index': 'metric'})
    all_eval_res = all_eval_res.melt(id_vars=['metric'], value_vars=['WordCountCosineSimilarity', 'TF_IDF', 'BM25', 'DirichletLM', 'PivotedNormalization', 'YourRanker'])
    all_eval_res = all_eval_res.rename(columns={'variable': 'ranker', 'value': 'score'})
    # draw a side by side bar plot
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(x="ranker", y="score", hue="metric", data=all_eval_res)
    plt.xticks(rotation=45)
    plt.savefig('relevance.png')


# TODO For the best-performing relevance function, calculate
# the MAP and NDCG scores for each query individually and plot these. Use the default
# hyperparameters in the code. Plot these scores on the y-axis and query number on the
# x-axis using a bar plot (i.e., one bar per query).

def test_best_rank_func():
    index_name = 'BasicInvertedIndexWithSPIMI'
    # index_name = 'OnDiskInvertedIndex'
    index_type = 'BasicInvertedIndexWithSPIMI'
    scorer_class = DirichletLM
    algorithm = SearchEngine(
        index_name, 
        dataset_path="wikipedia_1M_dataset.jsonl", 
        mwe_file_path='multi_word_expressions.txt',
        index_type=index_type,
        scorer_class=scorer_class,
    )
    relevance_df = pd.read_csv('relevance.csv')
    # 2. Run all of the dataset queries on the search algorithm.
    unique_query_df = relevance_df[['query', 'qid']].drop_duplicates()
    qid_to_query = {}
    for _, row in unique_query_df.iterrows():
        qid_to_query[row['qid']] = row['query']
    
    map_scores = []
    ndcg_scores = []
    
    for qid, query in tqdm(qid_to_query.items()):
        relevance_doc = relevance_df[relevance_df['qid'] == qid]
        doc_to_rel = {}
        for _, row in relevance_doc.iterrows():
            doc_to_rel[row['docid']] = row['rel']
        results = algorithm.search(query)
        result_rel = [doc_to_rel.get(result.docid, 0) for result in results]
        ideal_rel = sorted(doc_to_rel.values(), reverse=True)
        map_result_rel = list(map(lambda x: 1 if x > 0 else 0, result_rel))
        map_scores.append(map_score(map_result_rel))
        ndcg_scores.append(ndcg_score(result_rel, ideal_rel))
    
    plt.figure()
    plt.bar(range(len(map_scores)), map_scores)
    plt.savefig('map.png')
    plt.figure()
    plt.bar(range(len(ndcg_scores)), ndcg_scores)
    plt.savefig('ndcg.png')

if __name__ == '__main__':
    # NOTE: You can use this file on your command line by running 'python relevance.py'
    # test_all_rank_func()
    test_best_rank_func()