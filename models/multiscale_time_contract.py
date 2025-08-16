import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TC import temporal_contrast


class MultiScaleTimeContract(nn.Module):

    def __init__(self, seq_len, down_sampling_layers=2, down_sampling_method='avg', down_sampling_window=2, pred_len=8):
        super(MultiScaleTimeContract, self).__init__()

        self.down_sampling_method = down_sampling_method
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window

        final_length = max(int(seq_len / (2 ** 2 * 2)), 1)
        timesteps = pred_len if final_length > pred_len else final_length
        self.time_constrast_blocks = nn.ModuleList([temporal_contrast(timesteps=int(timesteps / (2 ** i))) for i in range(max(down_sampling_layers-1, 1))])

    def multi_scale_down_sampling(self, x_enc):
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        else:
            return x_enc

        x_enc_ori = x_enc

        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc)

        for i in range(self.down_sampling_layers - 1):
            x_enc_sampling = down_pool(x_enc_ori)

            if x_enc_sampling.shape[2] > 1:
                x_enc_sampling_list.append(x_enc_sampling)
            x_enc_ori = x_enc_sampling

        return x_enc_sampling_list

    def forward(self, x, target, model):
        x_enc_list = self.multi_scale_down_sampling(x)

        logits_list, scale_features_list = [], []
        for i, x_enc in zip(range(len(x_enc_list)), x_enc_list):
            logits, _, features = model(x_enc)
            features = F.normalize(features, dim=1)
            scale_features_list.append(features)
            logits_list.append(logits)

        multi_scale_logits = 0
        for i in range(len(logits_list)):
            multi_scale_logits += logits_list[i]
        multi_scale_loss = -torch.mean(torch.sum(F.log_softmax(multi_scale_logits, dim=1) * target, dim=1))

        neg_tcl_list = []
        if self.down_sampling_layers == 1:
            orin_neg_loss = self.time_constrast_blocks[0](scale_features_list[0], scale_features_list[0], pred_type='mid')
            neg_tcl_list.append(orin_neg_loss)
        # else:
        for i in range(len(scale_features_list) - 1):
            if scale_features_list[i+1].shape[2] > 1:
                neg_loss1 = self.time_constrast_blocks[i](scale_features_list[i], scale_features_list[i+1], pred_type='long')
                neg_tcl_list.append(neg_loss1)
            if scale_features_list[i].shape[2] > 1:
                neg_loss2 = self.time_constrast_blocks[i](scale_features_list[i+1], scale_features_list[i], pred_type='short')
                neg_tcl_list.append(neg_loss2)

        avg_ntcl_loss = sum(neg_tcl_list) / len(neg_tcl_list)

        return avg_ntcl_loss, multi_scale_loss
