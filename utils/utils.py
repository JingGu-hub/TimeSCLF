import datetime
import os
import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture

from cleanlab.internal.constants import EPSILON

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def create_file(path, filename, write_line=None, exist_create_flag=True):
    create_dir(path)
    filename = os.path.join(path, filename)

    if filename != None:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if not os.path.exists(filename):
            with open(filename, "a") as myfile:
                print("create new file: %s" % filename)
            with open(filename, "a") as myfile:
                myfile.write(write_line + '\n')
        elif exist_create_flag:
            new_file_name = filename + ".bak-%s" % nowTime
            os.system('mv %s %s' % (filename, new_file_name))
            with open(filename, "a") as myfile:
                myfile.write(write_line + '\n')

    return filename

def gmm_divide(args, loss):
    if args.use_gmm_divide_strategy:
        scaling_factor = float(max(np.median(loss), 100 * np.finfo(np.float64).eps))
        loss = np.exp(-1 * loss / max(scaling_factor, EPSILON))
        loss = 1 - loss

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(loss)
    prob = gmm.predict_proba(loss)

    thread = 0.5

    prob = prob[:, gmm.means_.argmin()]
    pred1 = (prob > thread)
    clean_ids = np.where(prob > thread)[0]
    u_ids = np.where(prob <= thread)[0]

    return pred1, prob, clean_ids, u_ids

def ood_gmm_divide(loss):
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(loss)
    prob = gmm.predict_proba(loss)

    prob = prob[:, gmm.means_.argmin()]
    pred1 = (prob > 0.5)
    u_ids = np.where(prob <= 0.5)[0]

    return pred1, prob, u_ids

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    # print(sim_weight.shape, sim_labels.shape)
    sim_weight = torch.ones_like(sim_weight)

    sim_weight = sim_weight / sim_weight.sum(dim=-1, keepdim=True)

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    # print(one_hot_label.shape)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    # print(pred_scores.shape)
    pred_labels = pred_scores.argmax(dim=-1)
    return pred_scores, pred_labels

def weighted_knn(cur_feature, feature, label, num_classes, knn_k=100, chunks=10, norm='global'):
    # distributed fast KNN and sample selection with three different modes
    num = len(cur_feature)
    num_class = torch.tensor([torch.sum(label == i).item() for i in range(num_classes)]).to(feature.device) + 1e-10
    pi = num_class / num_class.sum()
    split = torch.tensor(np.linspace(0, num, chunks + 1, dtype=int), dtype=torch.long).to(feature.device)
    score = torch.tensor([]).to(feature.device)
    pred = torch.tensor([], dtype=torch.long).to(feature.device)
    feature = torch.nn.functional.normalize(feature, dim=1)
    with torch.no_grad():
        for i in range(chunks):
            torch.cuda.empty_cache()
            part_feature = cur_feature[split[i]: split[i + 1]]

            part_score, part_pred = knn_predict(part_feature, feature.T, label, num_classes, knn_k)
            score = torch.cat([score, part_score], dim=0)
            pred = torch.cat([pred, part_pred], dim=0)

        # balanced vote
        if norm == 'global':
            # global normalization
            score = score / pi
        else:  # no normalization
            pass
        score = score/score.sum(1, keepdim=True)

    return score  # , pred

def divide_knn(feature_bank, labels, num_classes, ids=None):
    prediction_knn = weighted_knn(feature_bank, feature_bank, labels, num_classes, 200, 10)  # temperature in weighted KNN
    vote_y = torch.gather(prediction_knn, 1, labels.view(-1, 1)).squeeze()
    vote_max = prediction_knn.max(dim=1)[0]
    right_score = vote_y / vote_max

    pred = (right_score >= 1)

    return right_score, pred

def knn_scores(feature_bank, labels, num_classes):
    prediction_knn = weighted_knn(feature_bank, feature_bank, labels, num_classes, 200, 10)  # temperature in weighted KNN
    vote_y = torch.gather(prediction_knn, 1, labels.view(-1, 1)).squeeze()
    vote_max = prediction_knn.max(dim=1)[0]
    right_score = vote_y / vote_max

    return right_score

def compute_clean_ratio(loader, prob):
    clean_inds = np.where(prob > 0.5)[0].astype(int)
    clean_proportion = len(clean_inds) / float(len(loader.dataset))
    t = loader.dataset.target[clean_inds].detach().numpy() == loader.dataset.noisy_label[clean_inds].detach().numpy()
    pure_ratio = np.sum(loader.dataset.target[clean_inds].detach().numpy() == loader.dataset.noisy_label[clean_inds].detach().numpy()) / float(len(clean_inds))

    print('Clean proportion: %.4f, Pure ratio: %.4f' % (clean_proportion, pure_ratio))

def f1_scores(output, y_true):
    target_pred = torch.argmax(output.data, axis=1)

    y_true = y_true.detach().cpu().numpy()
    y_pred = target_pred.detach().cpu().numpy()

    # 计算F1分数
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    return f1_macro, f1_weighted, f1_micro

class CustomMultiStepLR:
    def __init__(self, optimizer, milestones, gammas):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gammas = gammas
        self.current_epoch = 0
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self):
        self.current_epoch += 1
        for i, milestone in enumerate(self.milestones):
            if self.current_epoch == milestone:
                self.optimizer.param_groups[0]['lr'] *= self.gammas[i]

def compute_similarity(features, temperature=0.05):
    # compute logits
    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
    anchor_dot_contrast[anchor_dot_contrast == float('inf')] = 1

    # for numerical stability
    pos_similarity_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    similarity = anchor_dot_contrast - pos_similarity_max.detach()
    similarity[similarity == float('inf')] = 1

    return similarity

def infoNCELoss(pos_features, neg_features, contrast_count=2, temperature=0.05):
    pos_batch_size = pos_features.shape[0] // contrast_count
    pos_mask = torch.eye(pos_batch_size, dtype=torch.float32).cuda()

    # tile mask
    pos_mask = pos_mask.repeat(contrast_count, contrast_count)

    # mask-out self-contrast cases
    general_mask = torch.scatter(
        torch.ones_like(pos_mask), 1,
        torch.arange(pos_batch_size * contrast_count).view(-1, 1).cuda(), 0
    )

    positive_mask = pos_mask * general_mask
    positive_mask[positive_mask==float('inf')] = 1
    poss = compute_similarity(pos_features, temperature=temperature)
    pos_similarity = positive_mask * poss

    negs = compute_similarity(neg_features, temperature=temperature)
    neg_similarity = positive_mask * negs

    # compute log_prob
    exp_logits = torch.exp(pos_similarity) + torch.exp(neg_similarity)
    log_prob = pos_similarity - torch.log(exp_logits.sum(1, keepdim=True))
    log_prob[log_prob==float('inf')] = 1

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = log_prob.sum(1) / positive_mask.sum(1)

    # loss
    loss = -1 * mean_log_prob_pos
    loss = loss.view(contrast_count, pos_batch_size).mean()

    return loss

def adjust_param(args, optimizer1, optimizer2):
    args.use_gmm_divide_strategy = True
    scheduler1 = CustomMultiStepLR(optimizer1, milestones=[30, 50, 100, 150], gammas=[0.05, 0.1, 0.1, 0.1])
    scheduler2 = CustomMultiStepLR(optimizer2, milestones=[30, 50, 100, 150], gammas=[0.05, 0.1, 0.1, 0.1])

    if args.dataset in ['ArticularyWordRecognition', 'HAR', 'UWaveGestureLibraryAll']:
        scheduler1 = CustomMultiStepLR(optimizer1, milestones=[100, 150], gammas=[0.5, 0.1])
        scheduler2 = CustomMultiStepLR(optimizer2, milestones=[100, 150], gammas=[0.5, 0.1])
    if args.dataset in ['HAR', 'FaceDetection', 'ElectricDeviceDetection', 'StarLightCurves', 'FordA']:
        args.use_gmm_divide_strategy = False


    return scheduler1, scheduler2
