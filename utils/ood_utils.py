import torch
import numpy as np
from cleanlab.outlier import OutOfDistribution
from cleanlab.rank import find_top_issues
from sklearn.neighbors import NearestNeighbors
from cleanlab.internal.constants import EPSILON

from utils.utils import divide_knn, gmm_divide, knn_scores, ood_gmm_divide


def compute_ood_distances(features):
    knn = NearestNeighbors(n_neighbors=min(10, features.shape[0] - 1), metric='cosine')
    knn.fit(features)
    k = knn.n_neighbors

    distances, indices = knn.kneighbors(features)

    avg_knn_distances = distances[:, :k].mean(axis=1)
    scaling_factor = float(max(np.median(avg_knn_distances), 100 * np.finfo(np.float64).eps))
    avg_knn_distances = np.exp(-1 * avg_knn_distances / max(scaling_factor, EPSILON))
    # avg_knn_distances = 1 - avg_knn_distances

    return avg_knn_distances

def obtain_ood_scores(features):
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    features = features[indices]

    avg_knn_distances = compute_ood_distances(features=features)
    # avg_knn_distances = (avg_knn_distances - np.min(avg_knn_distances)) / (np.max(avg_knn_distances) - np.min(avg_knn_distances))
    # avg_knn_distances = 1 - avg_knn_distances

    avg_knn_distances = avg_knn_distances[np.argsort(indices)]

    return avg_knn_distances

def build_ood_mask(args, epoch, features, labels, num_classes, indis_ids, ood_ids, clean_ids, u_ids):
    ratio_removal = int(0.2 * len(features))
    ood_mask = np.ones(len(labels))
    select_num, ood_num, id_num, clean_num = 0, 0, 0, 0
    devide_ood_ids = []

    ood_select_proportion, id_select_proportion, clean_select_proportion = np.array(0), np.array(0), np.array(0)
    if args.ood_noise_rate > 0 and args.ood_dispose_type != 'none':
        if args.ood_dispose_type == 'knn':
            train_ood_features_scores = obtain_ood_scores(features.detach().cpu().numpy())
            devide_ood_ids = find_top_issues(quality_scores=train_ood_features_scores, top=ratio_removal)
        elif args.ood_dispose_type == 'hl_gmm' and len(u_ids) > 1:
            train_ood_features_scores = obtain_ood_scores(features[u_ids].detach().cpu().numpy())
            feature_pred, feature_prob, devide_ood_ids = ood_gmm_divide(train_ood_features_scores.reshape(-1, 1))
            devide_ood_ids = u_ids[devide_ood_ids]
        elif args.ood_dispose_type == 'gmm':
            train_ood_features_scores = obtain_ood_scores(features.detach().cpu().numpy())
            feature_pred, feature_prob = ood_gmm_divide(train_ood_features_scores.reshape(-1, 1))
            devide_ood_ids = np.where(feature_prob <= 0.5)[0]
        if len(devide_ood_ids) != 0:
            ood_mask[devide_ood_ids] = 0

        # select_ood_proportion = np.isin(top_train_ood_features_idxs, ood_ids).sum() / len(ood_ids)
        # print(f"Selected OOD proportion: {select_ood_proportion}")

        ood_select_proportion = np.isin(devide_ood_ids, ood_ids).sum() / len(ood_ids)
        # avg_ood_scores = np.mean(train_ood_features_scores[np.setdiff1d(devide_ood_ids, ood_ids)])
        id_select_proportion = np.isin(devide_ood_ids, indis_ids).sum() / len(indis_ids)
        # avg_id_scores = train_ood_features_scores[np.setdiff1d(devide_ood_ids, indis_ids)]
        clean_select_proportion = np.isin(devide_ood_ids, clean_ids).sum() / len(clean_ids)

        ood_num = np.isin(devide_ood_ids, ood_ids).sum()
        id_num = np.isin(devide_ood_ids, indis_ids).sum()
        clean_num = np.isin(devide_ood_ids, clean_ids).sum()
        select_num = len(devide_ood_ids)

        # avg_ood_score = np.mean(train_ood_features_scores)
        # avg_true_ood_score = np.mean(train_ood_features_scores[ood_ids])
        # avg_false_ood_score = np.mean(train_ood_features_scores[no_ood_ids])
        #
        # gmm_no_ood_ids = np.setdiff1d(np.arange(len(features)), gmm_ood_ids)
        # gmm_avg_ood_score = np.mean(train_ood_features_scores[gmm_ood_ids])
        # gmm_avg_true_ood_score = np.mean(train_ood_features_scores[gmm_no_ood_ids])
        # gmm_ood_total_proportion = np.isin(gmm_no_ood_ids, ood_ids).sum() / len(ood_ids)
        # gmm_ood_proportion = np.isin(gmm_no_ood_ids, ood_ids).sum() / len(gmm_no_ood_ids)
        # print()
        #
        # pred, prob = divide_knn(torch.from_numpy(features).type(torch.float32).cuda(), labels, num_classes, ids=None)
        # clean_ids = torch.where(prob >= 1.0)[0].detach().cpu().numpy()
        # ood_mask = np.zeros(len(features))
        # ood_mask[clean_ids] = 1

    return ood_mask, ood_select_proportion, id_select_proportion, clean_select_proportion, select_num, ood_num, id_num, clean_num
