import math
import csv
from tqdm import tqdm
import pandas as pd
import numpy as np


# TODO (HW5): Implement NFaiRR
def nfairr_score(actual_omega_values: list[int], cut_off=200) -> float:
    """
    Computes the normalized fairness-aware rank retrieval (NFaiRR) score for a list of omega values
    for the list of ranked documents.
    If all documents are from the protected class, then the NFaiRR score is 0.

    Args:
        actual_omega_values: The omega value for a ranked list of documents
            The most relevant document is the first item in the list.
        cut_off: The rank cut-off to use for calculating NFaiRR
            Omega values in the list after this cut-off position are not used. The default is 200.

    Returns:
        The NFaiRR score
    """
    # TODO (HW5): Compute the FaiRR and IFaiRR scores using the given list of omega values
    # TODO (HW5): Implement NFaiRR
    considered_omega = actual_omega_values[:cut_off]
    ideal_omega = sorted(considered_omega, reverse=True)[:cut_off]
    weight_func = lambda omegas: omegas[0] + np.sum([omega / np.log2(pos + 2) for pos, omega in enumerate(omegas[1:])])

    nfairr = 0
    if len(considered_omega) > 0 and ideal_omega[0] != 0:
        ideal_fairr = weight_func(ideal_omega)
        actual_fairr = weight_func(considered_omega)
        nfairr = actual_fairr / ideal_fairr
    return nfairr


def my_nfaiir_score(nmag: np.array, j_a, cut_off=200) -> float:
    """
    Calculates the normalized fairness-aware rank retrieval (NFaiRR) score for a list of omega values

    Args:
        nmag: A numpy array of omega values for a ranked list of documents
        cut_off: The search result rank to stop calculating NFaiRR.
            The default cut-off is 200; calculate NFaiRR@200 to score your ranking function.
    """
    considered_nmag = nmag[:cut_off]
    num_classes = nmag.shape[0]
    weights = [2] + list(range(2, cut_off + 1))
    p_arr = 1 / np.log2(weights)
    Z = np.sum(p_arr)
    j_error = np.abs(np.sum(considered_nmag * p_arr, axis=1) / Z - j_a)
    return 1 - np.sum(j_error)


def map_score(search_result_relevances: list[int], cut_off=10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

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
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
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
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

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
        else:
            result_doc = [result[0] for result in result_doc]
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



# TODO (HW5): Implement NFaiRR metric for a list of queries to measure fairness for those queries
# NOTE: This has no relation to relevance scores and measures fairness of representation of classes
def run_fairness_test(attributes_file_path: str, protected_class: str, queries: list[str],
                      ranker, cut_off: int = 200) -> float:
    """
    Measures the fairness of the IR system using the NFaiRR metric.

    Args:
        attributes_file_path: The filename containing the documents about people and their demographic attributes
        protected_class: A specific protected class (e.g., Ethnicity, Gender)
        queries: A list containing queries
        ranker: A ranker configured with a particular scoring function to search through the document collection
        cut_off: The rank cut-off to use for calculating NFaiRR

    Returns:
        The average NFaiRR score across all queries
    """
    # TODO (HW5): Load person-attributes.csv
    person_attr_df = pd.read_csv(attributes_file_path)[['docid', protected_class]].dropna()
    # TODO (HW5): Find the documents associated with the protected class
    doc_to_attr = {docid: categoey for docid, categoey in zip(person_attr_df['docid'], person_attr_df[protected_class])}
    categories = set(doc_to_attr.values())
    score = []
    avg_score = 0

    # TODO (HW5): Loop through the queries and
    #       1. Create the list of omega values for the ranked list.
    #       2. Compute the NFaiRR score
    # NOTE: This fairness metric has some 'issues' (and the assignment spec asks you to think about it)
    for query in queries:
        result_doc = ranker.query(query)
        if isinstance(result_doc[0], dict):
            result_doc = [result.get('docid') for result in result_doc]
        else:
            result_doc = [result[0] for result in result_doc]
        avg_nfaiir = 0
        # if len(categories) > 0:
        #     for categoey in categories:
        #         actual_omega_values = [0 if doc_to_attr.get(doc, "") == categoey else 1 for doc in result_doc]
        #         avg_nfaiir += nfairr_score(actual_omega_values, cut_off)
        #     avg_nfaiir /= len(categories)
        actual_omega_values = [1 if doc_to_attr.get(doc, "") == "" else 0 for doc in result_doc]
        avg_nfaiir = nfairr_score(actual_omega_values, cut_off)
        score.append(avg_nfaiir)
    if len(score) > 0:
        avg_score = np.mean(score)
    return avg_score
    

if __name__ == "__main__":
    def one_hot_encode(indices, num_classes=None):
        if num_classes is None:
            num_classes = max(indices) + 1
        one_hot_arr = np.zeros((num_classes, len(indices)))
        one_hot_arr[indices, np.arange(len(indices))] = 1
        return one_hot_arr
    
    def arr_parser(arr, num_classes):
        one_hot_arr = one_hot_encode(arr, num_classes + 1)
        non_associated = (one_hot_arr[0] != 0)
        considered_arr = one_hot_arr[1:]
        j_a = 1 / num_classes
        considered_arr[:, non_associated] = j_a
        return considered_arr, tuple(non_associated.astype(int)), j_a

    # Order the nfairr scores by the protected class
    import itertools

    nfairr_order = []
    num_classes = 2
    arr_size = 3
    for arr in itertools.product(list(range(num_classes + 1)), repeat=arr_size):
        considered_arr, non_associated, j_a = arr_parser(arr, num_classes)
        nfairr_order.append((arr, my_nfaiir_score(considered_arr, j_a, arr_size), non_associated, nfairr_score(non_associated, arr_size)))
    nfairr_order = sorted(nfairr_order, key=lambda x: x[1], reverse=True)
    for arr, score, ref_arr, ref_score in nfairr_order:
        print(f"{arr} --> {score:.3f}")