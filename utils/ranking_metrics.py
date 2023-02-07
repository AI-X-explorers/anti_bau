import numpy as np

def dcg(scores):
    """
    Calculate DCG value based on the given score
    """
    dcg = 0
    for i in range(len(scores)):
        
        dcg += (np.power(2,scores[i],dtype=np.float128) - 1) / np.log(2+i)
    return dcg

def single_dcg(scores, i, j):
    """
    compute the single dcg that i-th element located j-th position
    :param scores:
    :param i:
    :param j:
    :return:
    """
    return (np.power(2, scores[i],dtype=np.float128) - 1) / np.log2(j+2)

def get_idcg(scores):
    """
    Calculate IDCG value based on the given score,the larger score means the better item
    """
    scores = sorted(scores)[::-1]
    return dcg(scores)

def ndcg(scores):
    """
    Calculate NDCG based on the given score
    """
    return dcg(scores)/get_idcg(scores)

def get_pairs(scores):
    """
    :param scores: given score list of documents for a particular query
    :return: the documents pairs whose firth doc has a higher value than second one.
    """
    pairs = []
    for i in range(len(scores)):
        for j in range(len(scores)):
            if scores[i] > scores[j]:
                pairs.append((i, j))
    return pairs


def compute_lambda(gt_scores, pred_scores, order_pairs):
    """
    gt_scores: the groundtruth score list of peptides
    pred_scores: the predicted score list of peptieds 
    orderd pairs: the partial oder pairs (i,j), where socre of peptide i > peptide j

    return:
        lambda value of each peptide
    """
    peptides_num = len(gt_scores)
    lambdas = np.zeros(peptides_num)
    IDCG = get_idcg(gt_scores)
    # Save dcg value
    dcg_values = {}
    for i, j in order_pairs:
        if (i, i) not in dcg_values:
            dcg_values[(i, i)] = single_dcg(gt_scores, i, i)
        if (j, j) not in dcg_values:
            dcg_values[(j, j)] = single_dcg(gt_scores, j, j)
        dcg_values[(i, j)] = single_dcg(gt_scores, i, j)
        dcg_values[(j, i)] = single_dcg(gt_scores, j, i)

    # Compute lambda
    for i, j in order_pairs:
        delta_ndcg = abs(dcg_values[(i,j)] + dcg_values[(j,i)] - dcg_values[(i,i)] - dcg_values[(j,j)])/IDCG # delta_ndcg
        rho = -1 / (1 + np.exp(pred_scores[i] - pred_scores[j]))
        lambdas[i] += rho * delta_ndcg
        lambdas[j] -= rho * delta_ndcg
    
    return lambdas
    
if __name__ == '__main__':
    scores = [0,1,4,2,3]
    print(dcg(scores))
    print(get_idcg(scores))