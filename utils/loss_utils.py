import numpy as np
import torch
import torch.nn.functional as F

def linear_rampup(args, current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, args, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(args, epoch, warm_up)

class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        features: Tensor of shape [2N, D], where D is the embedding dim.
                  The first N and the second N are positive pairs.
        """
        device = features.device
        batch_size = features.shape[0]
        assert batch_size % 2 == 0, "Batch size must be even."
        N = batch_size // 2

        # Normalize embeddings
        features = F.normalize(features, dim=1)

        # Compute cosine similarity matrix: [2N, 2N]
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Mask to remove similarity with self
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # Positive pairs: (i, i+N) and (i+N, i)
        pos_indices = torch.arange(N, device=device)
        positives = torch.cat([
            sim_matrix[pos_indices, pos_indices + N],
            sim_matrix[pos_indices + N, pos_indices]
        ], dim=0)

        # Denominator: sum over all similarities (excluding self)
        logits = sim_matrix
        labels = torch.zeros(2 * N, dtype=torch.long).to(device)

        # For numerical stability
        logits = torch.cat([
            positives.unsqueeze(1),
            logits
        ], dim=1)

        # Compute loss
        loss = F.cross_entropy(logits, labels)

        return loss

def compute_similarity(features, temperature=0.05):
    # compute logits
    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
    anchor_dot_contrast[anchor_dot_contrast == float('inf')] = 1

    # for numerical stability
    pos_similarity_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    similarity = anchor_dot_contrast - pos_similarity_max.detach()
    similarity[similarity == float('inf')] = 1

    return similarity

def infoNCE(pos_features, neg_features, contrast_count=2, temperature=0.05):
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
