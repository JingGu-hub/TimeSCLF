from __future__ import print_function
import sys

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import random
import os
import argparse
import numpy as np

from models.CNN import CNN
from models.PreResNet import *
from models.TC import temporal_contrast
from models.multiscale_time_contract import MultiScaleTimeContract

from utils.data_utils import build_dataset, load_loader
from utils.loss_utils import SemiLoss
from utils.ood_utils import build_ood_mask
from utils.scale_decom import scale_decom, ts_decom
from utils.tsmix_data_utils import tsmix_data

from utils.utils import gmm_divide, compute_clean_ratio, divide_knn, set_seed, create_file, CustomMultiStepLR, \
    f1_scores, adjust_param

from utils.constants import Multivariate2018_UEA30 as UEA_DATASET
from utils.constants import Four_dataset as OTHER_DATASET

import warnings
warnings.filterwarnings("ignore")


# Training
def train(args, epoch, net, net2, temporal_contr, optimizer, labeled_trainloader, unlabeled_trainloader, criterion, warm_up, ):
    net.train()
    net2.eval() #fix one network and train the other

    train_loss = 0
    all_logits_u = torch.zeros((len(unlabeled_trainloader.dataset), 128)).cuda()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, labels_x, w_x, index) in enumerate(labeled_trainloader):
        unlabeled_train_iter = iter(unlabeled_trainloader)
        inputs_u, u_index = next(unlabeled_train_iter)
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, labels_x, w_x = inputs_x.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u = inputs_u.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11, _, _ = net(inputs_u)
            outputs_u21, _, _ = net2(inputs_u)
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u21, dim=1)) / 2
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x, _, _ = net(inputs_x)
            
            px = torch.softmax(outputs_x, dim=1)
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
        all_targets = torch.cat([targets_x, targets_u], dim=0)

        logits, features, _ = net(all_inputs)
        logits_x = logits[:batch_size]
        logits_u = logits[batch_size:]
        all_logits_u[u_index] = features[batch_size:]

        Lx, Lu, lamb = criterion(args, logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        if args.use_time_constr:
            # temporal contrast loss
            avg_tcl_loss, multi_scale_loss = temporal_contr(inputs_x, all_targets[:batch_size], net)
            if args.use_multi_loss is not True:
                # loss = multi_scale_loss + lamb * Lu + penalty + 0.1 * temp_cont_loss
                loss = Lx + lamb * Lu + penalty + args.multi_loss_param * avg_tcl_loss
            elif args.use_tcloss is not True:
                loss = Lx + lamb * Lu + penalty + args.tc_param * multi_scale_loss
            else:
                loss = Lx + lamb * Lu + penalty + args.multi_loss_param * avg_tcl_loss + args.tc_param * multi_scale_loss
        else:
            loss = Lx + lamb * Lu + penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= num_iter

    return train_loss, all_logits_u

def warmup(net, dataloader, optimizer, CEloss, num_classes=10, mask=None):
    net.train()

    train_loss = 0
    features_all = torch.zeros([len(dataloader.dataset), 128]).cuda()
    pred_all = torch.zeros([len(dataloader.dataset), num_classes]).cuda()
    for i, (inputs, labels, index) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        mask_i = torch.from_numpy(mask[index]).cuda()

        optimizer.zero_grad()
        outputs, features, _ = net(inputs)
        loss = torch.mean(mask_i * CEloss(outputs, labels))

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)

        features_all[index] = features
        pred_all[index] = outputs
    train_loss /= len(dataloader.dataset)

    return train_loss, features_all, pred_all

def test(test_loader, net1, net2):
    net1.eval()
    net2.eval()

    correct, total, test_loss = 0, 0, 0
    f1_macro, f1_weighted, f1_micro, test_num = 0, 0, 0, 0
    for batch_idx, (inputs, targets, index) in enumerate(test_loader):
        with torch.no_grad():
            test_num += 1

            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1, _, _ = net1(inputs)
            outputs2, _, _ = net2(inputs)

            outputs = (outputs1+outputs2) / 2
            _, predicted = torch.max(outputs, 1)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()*targets.size(0)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

            f1_mac, f1_w, f1_mic = f1_scores(outputs, targets)
            f1_macro += f1_mac
            f1_weighted += f1_w
            f1_micro += f1_mic

    test_acc = 100.*correct/total
    test_loss /= total

    f1_macro, f1_weighted, f1_micro = f1_macro/test_num, f1_weighted/test_num, f1_micro/test_num

    return test_acc, test_loss, f1_macro, f1_weighted, f1_micro

def eval_train(args, eval_loader, model, CE, num_classes):
    model.eval()

    losses = torch.zeros(len(eval_loader.dataset)).cuda()
    features_all = torch.zeros([len(eval_loader.dataset), 128]).cuda()
    pred_all = torch.zeros([len(eval_loader.dataset), num_classes]).cuda()
    train_labels_all = torch.zeros(len(eval_loader.dataset)).type(torch.LongTensor).cuda()
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs, features, _ = model(inputs)
            loss = CE(outputs, targets)  

            losses[index]=loss
            features_all[index] = features
            pred_all[index] = outputs
            train_labels_all[index] = targets
    losses = (losses-losses.min())/(losses.max()-losses.min())

    if args.label_noise_rate==0.9: # average loss over last 5 epochs to improve convergence stability
        history = losses
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)

    return input_loss.reshape(-1,1).detach().cpu().numpy(), features_all, pred_all, train_labels_all

def main(archive='UEA', gpu_id=0, noise_type='symmetric', noise_rates=[0.5], corruption_dataset='InsectSound', ood_noise_rates=[0.], result_dir='./outputs/results/',
         ood_dispose_type='hl_gmm', use_time_constr=True, use_tcloss=True, use_multi_loss=True, warm_up_epoch=30, multiscale=3):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
    parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--id', default='')
    parser.add_argument('--seed', default=42)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--num_class', default=100, type=int)
    parser.add_argument('-model', type=str, default='CNN9layer', help='model type')  # 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101'

    parser.add_argument('--data_dir', type=str, default='../data/Multivariate2018_arff/Multivariate_arff',
                        help='dataset directory')
    parser.add_argument('--corruption_data_dir', type=str, default='../data/ts_noise_data/',
                        help='dataset directory')
    parser.add_argument('--result_dir', type=str, default='./outputs/results/', help='output directory')

    parser.add_argument('--dataset', default='ArticularyWordRecognition', type=str)
    parser.add_argument('--corruption_dataset', default='InsectSound', type=str)
    parser.add_argument('--noise_type', type=str, default='symmetric', help='symmetric, asymmetric, instance, pairflip')
    parser.add_argument('--label_noise_rate', type=float, default=0.5, help='label noise ratio')
    parser.add_argument('--ood_noise_rate', type=float, default=0, help='label noise ratio')

    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')

    parser.add_argument('--multi_loss_param', type=float, default=0.05, help='multi scale loss param')
    parser.add_argument('--tc_param', type=float, default=0.05, help='temporal contrast param')

    parser.add_argument('--use_time_constr', type=float, default=True, help='use_time_constr')
    parser.add_argument('--use_tcloss', type=float, default=True, help='use temporal contrast')
    parser.add_argument('--use_multi_loss', type=float, default=True, help='use_multi_loss')

    args = parser.parse_args()

    if archive == 'UEA':
        args.archive = archive
        datasets = UEA_DATASET
        args.data_dir = '../data/Multivariate2018_arff/Multivariate_arff/'
    elif archive == 'other':
        args.archive = archive
        datasets = OTHER_DATASET
        args.data_dir = '../data/ts_noise_data/'
    torch.cuda.set_device(gpu_id)
    args.result_dir, args.ood_dispose_type = result_dir, ood_dispose_type
    args.use_time_constr, args.use_tcloss, args.use_multi_loss = use_time_constr, use_tcloss, use_multi_loss

    # seeds = [1024]
    seeds = [2011, 2012, 2013, 2014, 2015]
    # seeds = [2048, 1024, 768, 384, 256, 2021, 2022, 2023]
    # seeds = [2023, 1024, 96, 384, 42]
    # seeds = [512, 128, 96, 64, 42]

    for dataset in datasets:
        args.dataset = dataset
        args.multiscale = 1 if args.dataset == 'PenDigits' else multiscale

        for noise_rate in noise_rates:
            for ood_noise_rate in ood_noise_rates:
                args.noise_type = noise_type
                args.label_noise_rate = noise_rate
                args.corruption_dataset = corruption_dataset
                args.ood_noise_rate = ood_noise_rate

                out_path = args.result_dir + args.dataset + '/'
                total_result_file = create_file(out_path, 'total_result.txt', 'statement,test_acc_list,test_acc,ood_select_proportion_list,ood_select_proportion,f1_macro_list,f1_macro,f1_weighted_list,f1_weighted,f1_micro_list,f1_micro,select_num,ood_num,id_num,clean_num', exist_create_flag=False)

                test_acc_list, test_loss_list, ood_select_proportion_list,  f1_macro_list, f1_weighted_list, f1_micro_list = [], [], [], [], [], []
                for seed in seeds:
                    args.seed = seed
                    set_seed(args)

                    result_file_name = '%s_%s_%s%.2f_ood%.2f.txt' % (args.model, args.dataset, args.noise_type, args.label_noise_rate, args.ood_noise_rate)
                    result_file = create_file(out_path, result_file_name, 'epoch,train_loss,test_loss,test_acc,ood_select_proportion,id_select_proportion,clean_select_proportion,f1_macro,f1_weighted,f1_micro')
                    train_loader, test_loader, train_dataset, train_aug_dataset, train_target, train_noisy_target, test_dataset, test_target, input_channel, seq_len, num_classes, indis_ids, ood_ids, clean_ids = build_dataset(args)
                    args.num_class = num_classes

                    print('| Building net')
                    net1 = CNN(input_channel=input_channel, n_outputs=num_classes).cuda()
                    net2 = CNN(input_channel=input_channel, n_outputs=num_classes).cuda()
                    temporal_contr1 = MultiScaleTimeContract(seq_len=seq_len, down_sampling_layers=args.multiscale).cuda()
                    temporal_contr2 = MultiScaleTimeContract(seq_len=seq_len, down_sampling_layers=args.multiscale).cuda()
                    cudnn.benchmark = True

                    criterion = SemiLoss()
                    optimizer1 = optim.SGD([{'params': net1.parameters()}, {'params': temporal_contr1.parameters()}], lr=args.lr, momentum=0.9, weight_decay=5e-4)
                    optimizer2 = optim.SGD([{'params': net2.parameters()}, {'params': temporal_contr2.parameters()}], lr=args.lr, momentum=0.9, weight_decay=5e-4)
                    scheduler1, scheduler2 = adjust_param(args, optimizer1, optimizer2)

                    CE = torch.nn.CrossEntropyLoss(reduction='none')

                    args.warm_up = warm_up_epoch
                    ood_mask1 = np.ones(len(train_dataset))
                    ood_mask2 = np.ones(len(train_dataset))
                    last_five_accs, last_five_losses, last_five_f1_macro, last_five_f1_weighted, last_five_f1_micro = [], [], [], [], []
                    ood_select_proportion, id_select_proportion, clean_select_proportion = np.array(0), np.array(0), np.array(0)
                    noise_num, select_num, ood_num, id_num, clean_num = 0, 0, 0, 0, 0
                    for epoch in range(args.num_epochs + 1):
                        scheduler1.step()
                        scheduler2.step()

                        input_loss1, features1, pred1, train_labels_all1 = eval_train(args, train_loader, net1, CE, num_classes)
                        input_loss2, features2, pred2, train_labels_all2 = eval_train(args, train_loader, net2, CE, num_classes)

                        if epoch < args.warm_up:
                            net1_train_loss, _, _ = warmup(net1, train_loader, optimizer1, CE, num_classes, ood_mask1)
                            net2_train_loss, _, _ = warmup(net2, train_loader, optimizer2, CE, num_classes, ood_mask2)
                        else:
                            pred1, prob1, clean_ids1, u_ids1 = gmm_divide(args, input_loss1)
                            pred2, prob2, clean_ids2, u_ids2 = gmm_divide(args, input_loss2)

                            # if epoch > args.warm_up:
                            ood_mask1, ood_select_proportion1, id_select_proportion1, clean_select_proportion1, select_num1, ood_num1, id_num1, clean_num1 = (
                                build_ood_mask(args, epoch, features1, train_labels_all1, num_classes, indis_ids, ood_ids, clean_ids, u_ids1))
                            ood_mask2, ood_select_proportion2, id_select_proportion2, clean_select_proportion2, select_num2, ood_num2, id_num2, clean_num2 = (
                                build_ood_mask(args, epoch, features2, train_labels_all2, num_classes, indis_ids, ood_ids, clean_ids, u_ids2))
                            ood_select_proportion = (ood_select_proportion1 + ood_select_proportion2) / 2
                            id_select_proportion = (id_select_proportion1 + id_select_proportion2) / 2
                            clean_select_proportion = (clean_select_proportion1 + clean_select_proportion2) / 2
                            clean_num = (len(clean_ids1) + len(clean_ids2)) / 2
                            noise_num = (len(u_ids1) + len(u_ids2)) / 2
                            select_num = (select_num1 + select_num2) / 2
                            ood_num = (ood_num1 + ood_num2) / 2
                            id_num = (id_num1 + id_num2) / 2

                            labeled_trainloader = load_loader(args, train_dataset, train_target, aug_dataset=train_aug_dataset, noisy_target=train_noisy_target, pred=pred1, prob=prob1, ood_mask=ood_mask1, ood_ids=ood_ids, mode='labeled')  # co-divide
                            unlabeled_trainloader = load_loader(args, train_dataset, train_target, aug_dataset=train_aug_dataset, noisy_target=train_noisy_target, pred=pred1, prob=prob1, ood_mask=ood_mask1, ood_ids=ood_ids, mode='unlabeled')  # co-divide
                            net1_train_loss, features1 = train(args, epoch, net1, net2, temporal_contr1, optimizer1, labeled_trainloader, unlabeled_trainloader, criterion, args.warm_up)  # train net1

                            labeled_trainloader = load_loader(args, train_dataset, train_target, aug_dataset=train_aug_dataset, noisy_target=train_noisy_target, pred=pred1, prob=prob1, ood_mask=ood_mask2, ood_ids=ood_ids, mode='labeled')  # co-divide
                            unlabeled_trainloader = load_loader(args, train_dataset, train_target, aug_dataset=train_aug_dataset, noisy_target=train_noisy_target, pred=pred2, prob=prob2, ood_mask=ood_mask2, ood_ids=ood_ids, mode='unlabeled')  # co-divide
                            net2_train_loss, features2 = train(args, epoch, net2, net1, temporal_contr2, optimizer2, labeled_trainloader, unlabeled_trainloader, criterion, args.warm_up)  # train net2

                        train_loss = (net1_train_loss + net2_train_loss) / 2
                        test_acc, test_loss, f1_macro, f1_weighted, f1_micro = test(test_loader, net1, net2)

                        print('Epoch:[%d/%d], train_loss:%.4f, test_loss:%.4f, test_acc:%.4f, ood_select_proportion:%.4f, id_select_proportion:%.4f, clean_select_proportion:%.4f, f1_macro:%.4f, f1_weighted:%.4f, f1_micro:%.4f, clean_num:%d, noise_num:%d, select_num:%d, ood_num:%d, id_num:%d, clean_num:%d' %
                              (epoch + 1, args.num_epochs, train_loss, test_loss, test_acc, ood_select_proportion, id_select_proportion, clean_select_proportion, f1_macro, f1_weighted, f1_micro, clean_num, noise_num, select_num, ood_num, id_num, clean_num))
                        with open(result_file, "a") as myfile:
                            myfile.write('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%d,%d\n' % (epoch + 1, train_loss, test_loss, test_acc, ood_select_proportion, id_select_proportion, clean_select_proportion, f1_macro, f1_weighted, f1_micro, clean_num, noise_num, select_num, ood_num, id_num, clean_num))

                        if (epoch + 5) >= args.num_epochs:
                            last_five_accs.append(test_acc)
                            last_five_losses.append(test_loss)
                            last_five_f1_macro.append(f1_macro)
                            last_five_f1_weighted.append(f1_weighted)
                            last_five_f1_micro.append(f1_micro)

                    test_accuracy = round(np.mean(last_five_accs), 4)
                    test_loss = round(np.mean(last_five_losses), 4)
                    f1_macro = round(np.mean(last_five_f1_macro), 4)
                    f1_weighted = round(np.mean(last_five_f1_weighted), 4)
                    f1_micro = round(np.mean(last_five_f1_micro), 4)
                    print('Test Accuracy:', test_accuracy, 'Test Loss:', test_loss, 'F1_macro:', f1_macro, 'F1_weighted:', f1_weighted, 'F1_micro:', f1_micro)

                    ood_select_proportion_list.append(round(ood_select_proportion.item(), 6))
                    test_acc_list.append(test_accuracy.item())
                    test_loss_list.append(test_loss.item())
                    f1_macro_list.append(f1_macro.item())
                    f1_weighted_list.append(f1_weighted.item())
                    f1_micro_list.append(f1_micro.item())

                test_acc, test_loss, avg_ood_prob = np.mean(test_acc_list), np.mean(test_loss_list), np.mean(ood_select_proportion_list)
                mean_f1_macro, mean_f1_weighted, mean_f1_micro = np.mean(f1_macro_list), np.mean(f1_weighted_list), np.mean(f1_micro_list)
                with open(total_result_file, "a") as myfile:
                    myfile.write('%s_%s_%s%.2f_ood%.2f,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (args.model, args.dataset, args.noise_type, args.label_noise_rate, args.ood_noise_rate,
                                 str(test_acc_list), test_acc, str(ood_select_proportion_list), avg_ood_prob, str(f1_macro_list), mean_f1_macro, str(f1_weighted_list), mean_f1_weighted, str(f1_micro_list), mean_f1_micro))

if __name__ == '__main__':
    # main(archive='UEA', gpu_id=0, noise_type='symmetric', noise_rates=[0.2], corruption_dataset='InsectSound', ood_noise_rates=[0], multiscale=2, result_dir='./outputs/results/')
    main(archive='other', gpu_id=0, noise_type='symmetric', noise_rates=[0.2], corruption_dataset='InsectSound', ood_noise_rates=[0], multiscale=2, result_dir='./outputs/results/')
    # main(archive='UEA', gpu_id=1, noise_type='symmetric', noise_rates=[0.5], corruption_dataset='InsectSound', ood_noise_rates=[0], multiscale=2, result_dir='./outputs/results/')
    # main(archive='other', gpu_id=1, noise_type='symmetric', noise_rates=[0.5], corruption_dataset='InsectSound', ood_noise_rates=[0], multiscale=2, result_dir='./outputs/results/')
    # main(archive='UEA', gpu_id=2, noise_type='instance', noise_rates=[0.4], corruption_dataset='InsectSound', ood_noise_rates=[0], multiscale=2, result_dir='./outputs/results/')
    # main(archive='other', gpu_id=2, noise_type='instance', noise_rates=[0.4], corruption_dataset='InsectSound', ood_noise_rates=[0], multiscale=2, result_dir='./outputs/results/')
    # main(archive='UEA', gpu_id=3, noise_type='pairflip', noise_rates=[0.4], corruption_dataset='InsectSound', ood_noise_rates=[0], multiscale=2, result_dir='./outputs/results/')
    # main(archive='other', gpu_id=3, noise_type='pairflip', noise_rates=[0.4], corruption_dataset='InsectSound', ood_noise_rates=[0], multiscale=2, result_dir='./outputs/results/')

