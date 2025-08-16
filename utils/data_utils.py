import os

import torch
import numpy as np
import torch.utils.data as data
import tsaug
from scipy.io.arff import loadarff
from torch.utils.data import DataLoader

from scipy import stats
import torch.nn.functional as F
from math import inf
import math

def transfer_labels(labels):
    indicies = np.unique(labels)
    num_samples = labels.shape[0]

    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indicies)[0][0]
        labels[i] = new_label

    return labels

def shuffler_dataset(x_train, y_train):
    indexes = np.array(list(range(x_train.shape[0])))
    np.random.shuffle(indexes)
    x_train = x_train[indexes]
    y_train = y_train[indexes]
    return x_train, y_train

def build_dataset_pt(args):
    data_path = args.data_dir + args.dataset
    train_dataset_dict = torch.load(os.path.join(data_path, "train.pt"))
    train_dataset = train_dataset_dict["samples"].numpy()  # (num_size, num_dimensions, series_length)
    train_target = train_dataset_dict["labels"].numpy()
    num_classes = len(np.unique(train_dataset_dict["labels"].numpy(), return_counts=True)[0])
    train_target = transfer_labels(train_target)

    test_dataset_dict = torch.load(os.path.join(data_path, "test.pt"))
    test_dataset = test_dataset_dict["samples"].numpy()  # (num_size, num_dimensions, series_length)
    test_target = test_dataset_dict["labels"].numpy()
    test_target = transfer_labels(test_target)

    if args.dataset == 'HAR':
        val_dataset_dict = torch.load(os.path.join(data_path, "val.pt"))
        val_dataset = val_dataset_dict["samples"].numpy()  # (num_size, num_dimensions, series_length)
        val_target = transfer_labels(val_dataset_dict["labels"].numpy())

        train_dataset = np.concatenate((train_dataset, val_dataset), axis=0)
        train_target = np.concatenate((train_target, val_target), axis=0)

    if args.dataset == 'FD':
        train_dataset = train_dataset.reshape(len(train_dataset), 1, -1)
        test_dataset = test_dataset.reshape(len(test_dataset), 1, -1)

    return train_dataset, train_target, test_dataset, test_target, num_classes

def build_dataset_uea(args, dataset_name=None):
    dataset = dataset_name if dataset_name is not None else args.dataset

    data_path = args.data_dir
    train_data = loadarff(os.path.join(data_path, dataset, dataset + '_TRAIN.arff'))[0]
    test_data = loadarff(os.path.join(data_path, dataset, dataset + '_TEST.arff'))[0]

    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([d.tolist() for d in t_data])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

    # def extract_data(data):
    #     res_data = []
    #     res_labels = []
    #     for t_data in data:
    #         t_data = np.array([d.tolist() for d in t_data])
    #         t_label = t_data[-1]
    #         t_data = t_data[:-1]
    #         t_data = np.char.decode(t_data, "utf-8").astype(float)
    #         t_label = t_label.decode("utf-8")
    #         res_data.append(t_data)
    #         res_labels.append(t_label)
    #
    #     res_data = np.expand_dims(np.array(res_data), axis=1)
    #     res_labels = np.array(res_labels)
    #     return res_data.swapaxes(1, 2), res_labels

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)

    train_dataset = train_X.transpose(0, 2, 1)
    train_target = train_y
    test_dataset = test_X.transpose(0, 2, 1)
    test_target = test_y

    num_classes = len(np.unique(train_target))
    train_target = transfer_labels(train_target)
    test_target = transfer_labels(test_target)

    ind = np.where(np.isnan(train_dataset))
    col_mean = np.nanmean(train_dataset, axis=0)
    col_mean[np.isnan(col_mean)] = 1e-6

    train_dataset[ind] = np.take(col_mean, ind[1])

    ind_test = np.where(np.isnan(test_dataset))
    test_dataset[ind_test] = np.take(col_mean, ind_test[1])

    train_dataset, train_target = shuffler_dataset(train_dataset, train_target)
    test_dataset, test_target = shuffler_dataset(test_dataset, test_target)

    return train_dataset, train_target, test_dataset, test_target, num_classes

def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std=0.1, seed=42, ood_ids=None):
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)

    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        t = W[y]
        A = x.reshape(1, -1).mm(W[y]).squeeze(0)

        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy() # if i not in ood_ids:

    l = [i for i in range(label_num)]
    new_label = [0 for i in range(len(labels))]
    for i in range(labels.shape[0]):
        if i not in ood_ids:
            new_label[i] = np.random.choice(l, p=P[i])
        else:
            new_label[i] = labels[i].item()
    # new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    print(f'noise rate = {(new_label != labels.detach().cpu().numpy()).mean()}')

    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1
        #
    print('****************************************')
    print('following is flip percentage:')

    for i in range(label_num):
        sum_i = sum(record[i])
        for j in range(label_num):
            if i != j:
                print(f"{record[i][j] / sum_i: .2f}", end='\t')
            else:
                print(f"{record[i][j] / sum_i: .2f}", end='\t')
        print()

    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            for j in range(label_num):
                print(f"{a[j]:.2f}", end="\t")
            print()
            cnt += 1
        if cnt >= 10:
            break
    print(P)
    return np.array(new_label)

def load_corruption_dataset(args):
    train_dataset = np.load(args.corruption_data_dir + '%s_train.npy' % args.corruption_dataset)
    test_dataset = np.load(args.corruption_data_dir + '%s_test.npy' % args.corruption_dataset)
    val_dataset = np.load(args.corruption_data_dir + '%s_val.npy' % args.corruption_dataset)

    return train_dataset, test_dataset, val_dataset

def split_evenly_3d(arr, n, axis=2):
    if arr.ndim != 3:
        raise ValueError("该函数仅适用于3维数组")

    size = arr.shape[axis]
    new_size = (size // n) * n  # 计算可以整除的长度

    if new_size == 0:
        raise ValueError(f"轴 {axis} 的大小太小，无法分成 {n} 段")

    # 用切片去除多余部分
    slicing = [slice(None)] * 3
    slicing[axis] = slice(0, new_size)
    trimmed = arr[tuple(slicing)]

    return np.split(trimmed, n, axis=axis)

def split_by_seq_len(arr, seq_len):
    if arr.ndim != 3:
        raise ValueError("该函数仅支持3维数组")

    a, b, c = arr.shape
    result = []

    # 只循环到完整长度，丢弃最后不满一段的部分
    for i in range(0, c - (c % seq_len), seq_len):
        chunk = arr[:, :, i:i+seq_len]
        result.append(chunk)

    return result


def group_time_series_numpy(x: np.ndarray, group_size: int = 10) -> np.ndarray:
    b, c, t = x.shape
    # assert c == 1, f"Expected shape (b, 1, T), got (b, {c}, {t})"
    # assert b >= group_size, "Batch size too small for requested group size"

    trimmed_len = b // group_size * group_size
    x = x[:trimmed_len]  # Trim extra samples
    x = x.reshape(-1, group_size, t)  # Reshape to (b // group_size, group_size, T)
    return x


def concat_time_axis_numpy(x: np.ndarray, group_size: int = 10) -> np.ndarray:
    b, c, t = x.shape

    trim_len = (b // group_size) * group_size
    x = x[:trim_len]  # Trim extra samples
    x = x.reshape(-1, group_size, c, t)  # (b//group, group_size, T)
    x = x.transpose(0, 2, 1, 3)  # (b//group, T, group_size)
    x = x.reshape(-1, c, group_size * t)  # (b//group, 1, group_size * T)

    return x

def adjust_corruption_dataset(args, train_dataset, channel, seq_len, ood_size):
    if train_dataset.shape[1] != 1:
        train_dataset = train_dataset.reshape(train_dataset.shape[0], 1, -1)

    corruption_channel = train_dataset.shape[1]
    time_steps = train_dataset.shape[2]

    channel_pad = math.ceil(channel / corruption_channel)
    ts_pad = math.ceil(seq_len / time_steps)
    if channel_pad > 1:
        # train_dataset = split_evenly_3d(train_dataset, channel_pad, axis=2)
        # train_dataset = np.concatenate(train_dataset, axis=1)
        train_dataset = group_time_series_numpy(train_dataset, group_size=math.ceil(channel / corruption_channel))

    if ts_pad > 1:
        train_dataset = concat_time_axis_numpy(train_dataset, group_size=ts_pad)

    train_dataset = split_by_seq_len(train_dataset, seq_len)
    train_dataset = np.concatenate(train_dataset, axis=0)
    dataset = train_dataset
    dataset = dataset[:, :channel, :]

    res_dataset = dataset.copy()
    size_pad = math.ceil(ood_size / len(dataset))
    if size_pad > 1:
        for i in range(size_pad):
            res_dataset = np.concatenate([res_dataset, dataset], axis=0)

    return res_dataset

def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    # x = x.cpu().numpy()
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            perm = np.random.permutation(len(splits))
            shuffled_splits = [splits[i] for i in perm]
            warp = np.concatenate(shuffled_splits).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return ret

def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def DataTransform(sample, max_seg=8, jitter_ratio=0.8):
    strong_aug = jitter(permutation(sample, max_segments=max_seg), jitter_ratio)
    return strong_aug

def flip_label(args, dataset, target, ratio):
    assert 0 <= ratio < 1

    _, channel, seq_len = dataset.shape
    target = np.array(target).astype(int)
    label = target.copy()
    n_class = len(np.unique(label))

    ood_ids = []
    if args.ood_noise_rate > 0:
        ood_ids = [i for i, t in enumerate(target) if np.random.random() < args.ood_noise_rate]
        if args.corruption_dataset == 'UrbanSound':
            train_dataset = np.load(args.corruption_data_dir + 'UrbanSound/%s_train_official.npy' % args.corruption_dataset)
            corruption_dataset = adjust_corruption_dataset(args, train_dataset, channel, seq_len, len(ood_ids))
            ood_data = corruption_dataset[np.random.permutation(np.arange(len(corruption_dataset)))[:len(ood_ids)]]
            ood_data = ood_data[np.random.permutation(len(ood_data))]
            # ood_data = tsaug.Reverse().augment(ood_data)
            # ood_data = tsaug.AddNoise(scale=1.0).augment(ood_data)
            # ood_data = tsaug.Crop(size=60).augment(ood_data)
            # ood_data = tsaug.TimeWarp(n_speed_change=3, max_speed_ratio=3.0).augment(ood_data)
        elif args.corruption_dataset == 'PhonemeSpectra':
            train_dataset = torch.load(os.path.join(args.corruption_data_dir, "%s/train.pt" % args.corruption_dataset))["samples"].numpy()
            test_dataset = torch.load(os.path.join(args.corruption_data_dir, "%s/test.pt" % args.corruption_dataset))["samples"].numpy()
            corruption_dataset = np.concatenate([train_dataset, test_dataset], axis=0)
            corruption_dataset = adjust_corruption_dataset(args, corruption_dataset, channel, seq_len, len(ood_ids))
            ood_data = corruption_dataset[np.random.permutation(np.arange(len(corruption_dataset)))[:len(ood_ids)]]
            ood_data = (ood_data - ood_data.min()) / (ood_data.max() - ood_data.min() + 1e-8)
        else:
            train_dataset = torch.load(os.path.join(args.corruption_data_dir, "%s/train.pt" % args.corruption_dataset))["samples"].numpy()
            test_dataset = torch.load(os.path.join(args.corruption_data_dir, "%s/test.pt" % args.corruption_dataset))["samples"].numpy()
            corruption_dataset = np.concatenate([train_dataset, test_dataset], axis=0)
            corruption_dataset = adjust_corruption_dataset(args, corruption_dataset, channel, seq_len, len(ood_ids))
            ood_data = corruption_dataset[np.random.permutation(np.arange(len(corruption_dataset)))[:len(ood_ids)]]
            # ood_data = tsaug.Reverse().augment(ood_data)
            # ood_data = tsaug.AddNoise(scale=1.0).augment(ood_data)
            # ood_data = (ood_data - ood_data.min()) / (ood_data.max() - ood_data.min() + 1e-8)

        dataset[ood_ids] = ood_data

    if args.noise_type == 'instance':
        # Instance
        num_classes = len(np.unique(target, return_counts=True)[0])
        data = torch.from_numpy(dataset).type(torch.FloatTensor)
        targets = torch.from_numpy(target).type(torch.FloatTensor).to(torch.int64)
        dataset_ = zip(data, targets)
        feature_size = dataset.shape[1] * dataset.shape[2]
        label = get_instance_noisy_label(n=ratio, dataset=dataset_, labels=targets, num_classes=num_classes,
                                         feature_size=feature_size, seed=args.seed, ood_ids=ood_ids)
    else:
        for i in range(label.shape[0]):
            if i not in ood_ids:
                # symmetric noise
                if args.noise_type == 'symmetric':
                    p1 = ratio / (n_class - 1) * np.ones(n_class)
                    p1[label[i]] = 1 - ratio
                    label[i] = np.random.choice(n_class, p=p1)
                elif args.noise_type == 'pairflip':
                    # pairflip
                    label[i] = np.random.choice([label[i], (target[i] + 1) % n_class], p=[1 - ratio, ratio])

    mask = np.array([int(x != y) for (x, y) in zip(target, label)])
    indis_ids = np.where(target != label)[0]
    clean_ids = np.setdiff1d(np.where(target == label)[0], ood_ids)

    return dataset, label, mask, indis_ids, ood_ids, clean_ids

def build_dataset(args):
    if args.archive == 'UEA':
        train_dataset, train_target, test_dataset, test_target, num_classes = build_dataset_uea(args)
    elif args.archive == 'other':
        train_dataset, train_target, test_dataset, test_target, num_classes = build_dataset_pt(args)
    input_channel = train_dataset.shape[1]
    seq_len = train_dataset.shape[2]

    # torch.save({'samples':torch.from_numpy(train_dataset).type(torch.FloatTensor), 'labels':torch.from_numpy(train_target).type(torch.LongTensor)}, os.path.join(f'../data/ts_noise_data/%s/train.pt' % args.dataset))
    # torch.save({'samples':torch.from_numpy(test_dataset).type(torch.FloatTensor), 'labels':torch.from_numpy(test_target).type(torch.LongTensor)}, os.path.join(f'../data/ts_noise_data/%s/test.pt' % args.dataset))

    print('dataset', args.dataset, "train_dataset shape = ", train_dataset.shape, ", test_dataset shape = ", test_dataset.shape)

    # corrupt label
    train_noisy_target, ood_ids = train_target.copy(), []
    if args.label_noise_rate > 0 or args.ood_noise_rate > 0:
        train_dataset, train_noisy_target, mask_train_target, indis_ids, ood_ids, clean_ids = flip_label(args=args, dataset=train_dataset, target=train_target, ratio=args.label_noise_rate)

    train_aug_dataset = DataTransform(train_dataset)

    # load train_loader
    train_loader = load_loader(args, train_dataset, train_target, aug_dataset=train_aug_dataset, noisy_target=train_noisy_target, mode='train', ood_ids=ood_ids)
    # load test_loader
    test_loader = load_loader(args, test_dataset, test_target, shuffle=False, mode='test', ood_ids=ood_ids)

    return train_loader, test_loader, train_dataset, train_aug_dataset, train_target, train_noisy_target, test_dataset, test_target, input_channel, seq_len, num_classes, indis_ids, ood_ids, clean_ids

class TimeDataset(data.Dataset):
    def __init__(self, dataset, target=None, aug_dataset=None, noisy_label=None, pred=None, prob=None, ood_mask=None, ood_ids=None, mode='train'):
        self.dataset = dataset
        # self.dataset = np.expand_dims(self.dataset, 1)
        # print("dataset shape = ", dataset.shape)
        if len(self.dataset.shape) == 2:
            self.dataset = torch.unsqueeze(self.dataset, 1)
        # print("dataset shape = ", self.dataset.shape)
        self.target = target
        self.noisy_label = noisy_label
        self.ood_mask = ood_mask
        self.ood_ids = ood_ids

        self.mode = mode

        if mode == 'labeled':
            self.pred_idx = (pred * ood_mask).nonzero().reshape(-1)
            self.probability = [prob[i] for i in self.pred_idx]
            self.dataset = dataset[self.pred_idx]
            self.aug_dataset = aug_dataset[self.pred_idx]
            self.noisy_label = [noisy_label[i] for i in self.pred_idx]
        elif mode == 'unlabeled':
            self.pred = pred
            self.pred_idx = ((1 - pred) * ood_mask).nonzero().reshape(-1)
            self.dataset = dataset[self.pred_idx]
            self.aug_dataset = aug_dataset[self.pred_idx]
            self.noisy_label = [noisy_label[i] for i in self.pred_idx]

        # if len(self.dataset.data) == 0:
        #     self.dataset = dataset[0].unsqueeze(0)
        #     self.aug_dataset = aug_dataset[0].unsqueeze(0)
        #     self.noisy_label = [noisy_label[0]]

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.dataset[index], self.noisy_label[index], index
        elif self.mode == 'test':
            return self.dataset[index], self.target[index], index
        elif self.mode == 'labeled':
            return self.dataset[index], self.noisy_label[index], self.probability[index], index
        elif self.mode == 'unlabeled':
            return self.dataset[index], index

    def __len__(self):
        return len(self.dataset)

def load_loader(args, data, target, aug_dataset=None, noisy_target=None, pred=None, prob=None, shuffle=True, mode='train', ood_mask=None, ood_ids=None):
    if mode == 'train':
        dataset = TimeDataset(dataset=torch.from_numpy(data).type(torch.FloatTensor),
                              target=torch.from_numpy(target).type(torch.LongTensor),
                              noisy_label=torch.from_numpy(noisy_target).type(torch.LongTensor), ood_ids=ood_ids, mode='train')
    elif mode == 'test':
        dataset = TimeDataset(dataset=torch.from_numpy(data).type(torch.FloatTensor),
                              target=torch.from_numpy(target).type(torch.LongTensor), ood_ids=ood_ids, mode='test')
    elif mode == 'labeled':
        dataset = TimeDataset(dataset=torch.from_numpy(data).type(torch.FloatTensor),
                              target=torch.from_numpy(target).type(torch.LongTensor),
                              aug_dataset=torch.from_numpy(aug_dataset).type(torch.FloatTensor),
                              noisy_label=torch.from_numpy(noisy_target).type(torch.LongTensor),
                              pred=torch.from_numpy(pred).type(torch.LongTensor),
                              prob=torch.from_numpy(prob).type(torch.FloatTensor), ood_mask=ood_mask, ood_ids=ood_ids, mode='labeled')
    elif mode == 'unlabeled':
        dataset = TimeDataset(dataset=torch.from_numpy(data).type(torch.FloatTensor),
                              target=torch.from_numpy(target).type(torch.LongTensor),
                              aug_dataset=torch.from_numpy(aug_dataset).type(torch.FloatTensor),
                              noisy_label=torch.from_numpy(noisy_target).type(torch.LongTensor),
                              pred=torch.from_numpy(pred).type(torch.LongTensor),
                              prob=torch.from_numpy(prob).type(torch.FloatTensor), ood_mask=ood_mask, ood_ids=ood_ids, mode='unlabeled')
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=shuffle)

    return loader
