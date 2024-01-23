import numpy as np
import pandas as pd
from tqdm import tqdm


def map_score(search_result_relevances: list[int], cut_off=10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_results: A list of 0/1 values for whether each search result returned by your 
                        ranking function is relevant
        cut_off: The search result rank to stop calculating MAP. The default cut-off is 10;
                 calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    # TODO: Implement MAP
    all_rel_doc = np.sum([1 if rel > 0 else 0 for rel in search_result_relevances])
    correct_pos = [pos for pos in range(cut_off) if search_result_relevances[pos] > 0]
    precision_k = [(i + 1) / (pos + 1) for i, pos in enumerate(correct_pos)]
    return np.sum(precision_k) / all_rel_doc if len(precision_k) > 0 else 0


def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_off=10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: 
            A list of relevance scores for the results returned by your ranking function in the
            order in which they were returned. These are the human-derived document relevance scores,
            *not* the model generated scores.
            
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score in descending order.
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    # TODO: Implement NDCG
    actual_len = min(len(search_result_relevances), cut_off)
    dcg = search_result_relevances[0] + np.sum([gain / np.log2(pos + 2) for pos, gain in enumerate(search_result_relevances[1:actual_len])]) if actual_len != 0 else 0
    ideal_len = min(len(ideal_relevance_score_ordering), cut_off)
    idcg = ideal_relevance_score_ordering[0] +np.sum([gain / np.log2(pos + 2) for pos, gain in enumerate(ideal_relevance_score_ordering[1:ideal_len])]) if ideal_len != 0 else 0
    return dcg / idcg if idcg > 0 else 0


def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename [str]: The filename containing the relevance data to be loaded

        ranker: A ranker configured with a particular scoring function to search through the document collection.
                This is probably either a Ranker or a L2RRanker object, but something that has a query() method

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # TODO: Load the relevance dataset.
    relevance_df = pd.read_csv(relevance_data_filename)
    # TODO. Run all the dataset's queries using the provided ranker object.
    map_scores = []
    ndcg_scores = []

    # Label the outputs from the algorithm using the relevance dataset.
    # NOTE: Our relevance score has five levels (1-5). Use multi-level scores (1-5) when calculating NDCG.
    # When calculating MAP, treat 1-3 as non-relevant (0) and 4-5 as relevant (1) because MAP uses binary scores.
    # For example, if we have a list of labeled outputs like [4, 5, 3, 4, 1, 2, 1],
    # convert this into [1, 1, 0, 1, 0, 0, 0] when you calculate the MAP score.
    for query, relevance_doc in tqdm(relevance_df.groupby('query')):
        doc_to_rel = {}
        for _, row in relevance_doc.iterrows():
            doc_to_rel[row['docid']] = row['rel']
        result_doc = ranker.query(query)
        if isinstance(result_doc[0], dict):
            result_doc = [result.get('docid') for result in result_doc]
        result_rel = [doc_to_rel.get(result, 1) for result in result_doc]
        ideal_rel = sorted(doc_to_rel.values(), reverse=True)
        map_result_rel = list(map(lambda x: 1 if x > 3 else 0, result_rel))
        map_scores.append(map_score(map_result_rel))
        ndcg_scores.append(ndcg_score(result_rel, ideal_rel))
    # TODO: Compute the average MAP and NDCG across all queries and return the scores. 
    map_avg_score = np.mean(map_scores)
    ndcg_avg_score = np.mean(ndcg_scores)
    # 3: Return the scores.
    return {'map': map_avg_score, 'ndcg': ndcg_avg_score, 'map_scores': map_scores, 'ndcg_scores': ndcg_scores}
