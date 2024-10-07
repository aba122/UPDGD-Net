import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os.path as osp
import utils
from utils import AverageMeter
import MLdataset
import argparse
import time
from model import get_model
import evaluation
import torch
import numpy as np
import myloss
from myloss import Loss
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
import copy
from torch.optim.lr_scheduler import StepLR
import numpy as np

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Uncertainty(model, T, loader):
    model.train()
    total_Uncertainty = []
    total_Mean_predictions = []
    total_orig_indices = []

    for i, (data, label, inc_V_ind, inc_L_ind, orig_indices) in enumerate(loader):
        data = [v_data.to('cuda:0') for v_data in data]
        inc_V_ind = inc_V_ind.float().to('cuda:0')

        sum_predictions = 0
        sum_sq_predictions = 0

        for t in range(T):
            datat = [torch.clone(v_data).detach() for v_data in data]
            inc_V_indt = torch.clone(inc_V_ind).detach()

            pred, _, _, _ = model(datat, mask=inc_V_indt)
            pred = pred[0].detach().cpu()

            sum_predictions += pred
            sum_sq_predictions += pred.pow(2)

            del datat, inc_V_indt, pred

        Mean_predictions = sum_predictions / T
        Uncertainty = (sum_sq_predictions / T) - (Mean_predictions.pow(2))

        total_Mean_predictions.append(Mean_predictions)
        total_Uncertainty.append(Uncertainty)
        total_orig_indices.append(orig_indices)

        del sum_predictions, sum_sq_predictions

        torch.cuda.empty_cache()

    total_Mean_predictions = torch.cat(total_Mean_predictions, dim=0)
    total_Uncertainty = torch.cat(total_Uncertainty, dim=0)
    total_orig_indices = torch.cat(total_orig_indices, dim=0)

    min_val = total_Uncertainty.min(dim=0, keepdim=True).values
    max_val = total_Uncertainty.max(dim=0, keepdim=True).values
    Uncertainty_normalized = (total_Uncertainty - min_val) / (max_val - min_val)

    return Uncertainty_normalized, total_Mean_predictions, total_orig_indices

def pseudo_label_indicator(Uncertainty, inverted_inc_L_ind, ratio):
    
    inverted_inc_L_ind = torch.tensor(inverted_inc_L_ind)
    flat_uncertainty = Uncertainty.flatten()
    flat_inverted_ind = inverted_inc_L_ind.flatten()

    valid_uncertainties = flat_uncertainty[flat_inverted_ind.bool()]

    num_samples = int(ratio * valid_uncertainties.size(0))

    num_samples = min(num_samples, valid_uncertainties.size(0))

    _, selected_indices = torch.topk(valid_uncertainties, num_samples, largest=False, sorted=True)

    G = torch.zeros_like(inverted_inc_L_ind)

    selected_indices_original = torch.nonzero(flat_inverted_ind)[selected_indices].squeeze()

    G.view(-1)[selected_indices_original] = 1

    return G

def Dselect(prediction, Uncertainity, Kp):
    Gp = torch.where(Uncertainity < Kp, 1, 0)
    Ds = prediction * Gp
    return Ds, Gp

def regularize_loss(param, lambda_reg):
    return lambda_reg * torch.sum(param ** 2)

def train(loader, model, loss_model,opt, sche, epoch, logger, predict_Y,G,origin_inc_L_ind):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    train_predictions = []
    for i, (data, label, inc_V_ind, inc_L_ind,_) in enumerate(loader):
        ori_inc_L_ind = origin_inc_L_ind[128*i:(i+1)*128,:]
        ori_inverted_inc_L_ind = np.logical_not(ori_inc_L_ind)
        ori_inc_L_ind = torch.tensor(ori_inc_L_ind).to('cuda:0')
        ori_inverted_inc_L_ind = torch.tensor(ori_inverted_inc_L_ind).to('cuda:0')
        predict = predict_Y[128*i:(i+1)*128,:]
        g = G[128*i:(i+1)*128,:].to('cuda:0')

        data_time.update(time.time() - end)
        data = [v_data.to('cuda:0') for v_data in data]
        inc_V_ind2 = inc_V_ind.unsqueeze(-1).to('cuda:0')
        try2 = inc_V_ind2[:,2,:].detach().cpu().numpy()
        for i in range(5):
            data[i] = data[i].mul(inc_V_ind2[:,i,:])

        weighted_inverted_inc_L_ind = ori_inverted_inc_L_ind  * g

        label = label.to('cuda:0')
        data0 = data[0].detach().cpu().numpy()
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        inc_L_ind = inc_L_ind.float().to('cuda:0')

        inverted_inc_L_ind = torch.logical_not(inc_L_ind).int()

        pred, x_bar_list, x_tran,cls_tokens = model(data, mask=inc_V_ind)

        train_predictions.append(pred[0].detach().cpu())

        cont_loss2, sim, labels1 = loss_model.contrastive_loss2(x_tran, label.cuda(), inc_V_ind.cuda(), inc_L_ind.cuda(),predict.cuda(),epoch)

        cont_loss3 = loss_model.contrastive_loss3(label.cuda(),cls_tokens.cuda(),inc_L_ind.cuda())

        cls_loss1 =loss_model.weighted_BCE_loss(pred[0], label.cuda(), inc_L_ind.cuda())
        cls_loss2 = loss_model.weighted_BCE_loss(pred[0], label.cuda(), weighted_inverted_inc_L_ind.cuda())

        cls_loss = cls_loss1 + cls_loss2
        loss = cls_loss + args.alpha * cont_loss2 + args.beta * cont_loss3

        opt.zero_grad()
        loss.backward()
        if isinstance(sche, CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))

        opt.step()
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

    if isinstance(sche, StepLR):
        sche.step()
    logger.info('Epoch:[{0}]\t'
                'Time {batch_time.avg:.3f}\t'
                'Data {data_time.avg:.3f}\t'
                'Loss {losses.avg:.3f}\t'
                'Construct_Loss {con_loss:.3f}\t'
                'Regl_loss {reg_loss:.3f}'.format(
        epoch, batch_time=batch_time,
        data_time=data_time, losses=losses,con_loss=cls_loss1,reg_loss=cont_loss3))

    train_predictions = torch.cat(train_predictions, dim=0)

    return losses, model, train_predictions

def test(loader, model, loss_model, epoch,logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_labels = []
    total_preds = []
    model.eval()
    end = time.time()
    for i, (data, label, inc_V_ind, inc_L_ind,_) in enumerate(loader):
        # data_time.update(time.time() - end)
        data=[v_data.to('cuda:0') for v_data in data]
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        pred, _, _,_ = model(data,mask=inc_V_ind)
        pred = pred[0].cpu()
        total_labels = np.concatenate((total_labels,label.numpy()),axis=0) if len(total_labels)>0 else label.numpy()
        total_preds = np.concatenate((total_preds,pred.detach().numpy()),axis=0) if len(total_preds)>0 else pred.detach().numpy()

        batch_time.update(time.time()- end)
        end = time.time()
    total_labels=np.array(total_labels)
    total_preds=np.array(total_preds)

    evaluation_results=evaluation.do_metric(total_preds,total_labels)
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'AP {ap:.3f}\t'
                  'HL {hl:.3f}\t'
                  'RL {rl:.3f}\t'
                  'AUC {auc:.3f}\t'.format(
                        epoch,   batch_time=batch_time,
                        ap=evaluation_results[0],
                        hl=evaluation_results[1],
                        rl=evaluation_results[2],
                        auc=evaluation_results[3]
                        ))
    return evaluation_results


def main(args,file_path):
    data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view.mat')
    fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view_MaskRatios_' + str(
                                args.mask_view_ratio) + '_LabelMaskRatio_' +
                                str(args.mask_label_ratio) + '_TraindataRatio_' +
                                str(args.training_sample_ratio) + '.mat')

    print(fold_data_path)
    folds_num = args.folds_num
    folds_results = [AverageMeter() for i in range(9)]
    if args.logs:
        logfile = osp.join(args.logs_dir,args.name+args.dataset+'_V_' + str(
                                    args.mask_view_ratio) + '_L_' +
                                    str(args.mask_label_ratio) + '_T_' +
                                    str(args.training_sample_ratio) + '_'+str(args.alpha)+'_'+str(args.beta)+'.txt')
    else:
        logfile=None
    logger = utils.setLogger(logfile)
    device = torch.device('cuda:0')
    res_list_val = []
    for fold_idx in range(folds_num):
        fold_idx = fold_idx
        train_dataloder, train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,
                                                                    training_ratio=args.training_sample_ratio,
                                                                    fold_idx=fold_idx, mode='train', batch_size=128,
                                                                    shuffle=False, num_workers=0)
        test_dataloder, test_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,
                                                                  training_ratio=args.training_sample_ratio,
                                                                  val_ratio=0.15, fold_idx=fold_idx, mode='test',
                                                                  batch_size=128, num_workers=0)
        val_dataloder, val_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,
                                                                training_ratio=args.training_sample_ratio,
                                                                fold_idx=fold_idx, mode='val', batch_size=128,
                                                                num_workers=0)
        d_list = train_dataset.d_list
        classes_num = train_dataset.classes_num

        model = get_model(len(d_list), d_list, d_model=512, n_layers=6, heads=4, classes_num=train_dataset.classes_num,
                          dropout=0.3, exponent=2)
        # print(model)
        loss_model = Loss()
        # crit = nn.BCELoss()
        optimizer = SGD(model.parameters(), lr=0.02, momentum=0.9)
        # optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=200, gamma=0.90)
        logger.info('train_data_num:' + str(len(train_dataset)) + '  test_data_num:' + str(
            len(test_dataset)) + '   fold_idx:' + str(fold_idx))
        print(args)
        static_res = 0
        epoch_results = [AverageMeter() for i in range(9)]
        total_losses = AverageMeter()
        train_losses_last = AverageMeter()
        best_epoch = 0
        best_model_dict = {'model': model.state_dict(), 'epoch': 0, 'Val_AP': 0.0}
        predict_Y = torch.tensor(train_dataloder.dataset.cur_labels)
        best_prediction = []
        origin_inc_L_ind = train_dataset.cur_inc_L_ind
        inverted_inc_L_ind = np.logical_not(origin_inc_L_ind)
        Origin_label = train_dataloder.dataset.cur_labels
        train_losses_per_epoch = []
        val_ap_per_epoch = []
        initial_ratio = 0.01
        final_ratio = 1.0
        growth_rate = 1 / args.epochs
        ratio = initial_ratio
        T = args.T
        G = torch.zeros_like(torch.tensor(inverted_inc_L_ind))
        for epoch in range(args.epochs):
            train_losses, model, predict_Y = train(train_dataloder, model.cuda(), loss_model,  optimizer,
                                                   scheduler, epoch, logger, predict_Y, G, origin_inc_L_ind)
            val_results = test(val_dataloder, model, loss_model, epoch, logger)
            if val_results[0] * 0.25 + val_results[2] * 0.25 + val_results[3] * 0.5 >= static_res:
                static_res = val_results[0] * 0.25 + val_results[2] * 0.25 + val_results[3] * 0.5
                best_model_dict['model'] = copy.deepcopy(model.state_dict())
                best_model_dict['epoch'] = epoch
                best_model_dict['Val-AP'] = val_results[0]
                best_epoch = epoch
                best_prediction = predict_Y
            train_losses_last = train_losses
            total_losses.update(train_losses.sum)

            train_losses_per_epoch.append(train_losses.avg)
            val_ap_per_epoch.append(val_results[0])

            ratio += growth_rate

            Un, Psuedo_labels, _ = Uncertainty(model, T, train_dataloder)


            G = pseudo_label_indicator(Un, inverted_inc_L_ind, ratio)

            G = G * torch.tensor(inverted_inc_L_ind)

            train_dataloder.dataset.cur_labels = Origin_label + np.multiply(Psuedo_labels.numpy(), inverted_inc_L_ind)

        print(next(model.parameters()).device)
        model.load_state_dict(best_model_dict['model'])
        test_results = test(test_dataloder, model, loss_model, epoch, logger)
        best_val_results = test(val_dataloder, model, loss_model, best_model_dict['epoch'], logger)
        res_list_val.append(best_val_results[0])

        logger.info(
            'final: fold_idx:{} best_epoch:{}\t best:ap:{:.4}\t HL:{:.4}\t RL:{:.4}\t AUC_me:{:.4}\n'.format(fold_idx,
                                                                                                             best_epoch,
                                                                                                             test_results[
                                                                                                                 0],
                                                                                                             test_results[
                                                                                                                 1],
                                                                                                             test_results[
                                                                                                                 2],
                                                                                                             test_results[
                                                                                                                 3]))

        for i in range(9):
            folds_results[i].update(test_results[i])

        if args.save_curve:
            np.save(osp.join(args.curve_dir, args.dataset + '_V_' + str(args.mask_view_ratio) + '_L_' + str(
                args.mask_label_ratio)) + '_' + str(fold_idx) + '.npy',
                    np.array(list(zip(epoch_results[0].vals, train_losses.vals))))

    file_handle = open(file_path, mode='a')
    if os.path.getsize(file_path) == 0:
        file_handle.write(
            'A AP HL RL AUCme one_error coverage macAUC macro_f1 micro_f1 lr alpha beta gamma\n')
    # generate string-result of 9 metrics and two parameters
    # (folds_results[0].vals)
    res_list = [str(round(res.avg, 4)) + '+' + str(round(res.std, 4)) for res in folds_results]
    res_list.extend([str(args.lr), str(args.alpha), str(args.beta), str(args.gamma)])
    res_str = ' '.join(res_list)
    print(res_str)
    file_handle.write('\n')
    file_handle.close()


def filterparam(file_path,index):
    params = []
    if os.path.exists(file_path):
        file_handle = open(file_path, mode='r')
        lines = file_handle.readlines()
        lines = lines[1:] if len(lines)>1 else []
        params = [[float(line.split(' ')[idx]) for idx in index] for line in lines ]
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__)) 
    parser.add_argument('--logs-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'records'))
    parser.add_argument('--file-path', type=str, metavar='PATH', 
                        default='')
    parser.add_argument('--root-dir', type=str, metavar='PATH',
                        default='/home/xiewulin/UPDGD/data/')
    parser.add_argument('--dataset', type=str, default='')#mirflickr corel5k pascal07 iaprtc12 espgame
    parser.add_argument('--datasets', type=list, default=['corel5k'])
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--mask-label-ratio', type=float, default=0.5)
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=1, type=int)
    parser.add_argument('--weights-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'curves'))
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    
    parser.add_argument('--name', type=str, default='10_final_')
    # Optimization args
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=500)
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1e-1)
    parser.add_argument('--beta', type=float, default=1e-1)
    parser.add_argument('--T', type=float, default=5)


    
    args = parser.parse_args()
    
    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if args.save_curve:
        if not os.path.exists(args.curve_dir):
            os.makedirs(args.curve_dir)
    if True:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)
    lr_list = [1e-1,1e-5,1e0]
    alpha_list = [1e1]
    beta_list = [1e-1]
    T_list = [5]
    Mask_view_ratio_list = [0.5]

    
    for lr in lr_list:
        args.lr = lr
        if args.lr >= 0.01:
            args.momentumkl = 0.90
        for alpha in alpha_list:
            args.alpha = alpha
            for beta in beta_list:
                args.beta = beta
                for T in T_list:
                    args.T = T
                    for dataset in args.datasets:
                        args.dataset = dataset
                        file_path = osp.join(args.records_dir,args.name+args.dataset+'_ViewMask_' + str(
                                        args.mask_view_ratio) + '_LabelMask_' +
                                        str(args.mask_label_ratio) + '_Training_' + 
                                        str(args.training_sample_ratio) + '_bs128.txt')
                        args.file_path = file_path
                        existed_params = filterparam(file_path,[-3,-2,-1])
                        main(args,file_path)

        
    
    
