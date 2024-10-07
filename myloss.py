import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def complete_similarity_matrix(sim, mask_v):
    num_views, num_samples, _ = sim.size()

    valid_mask = mask_v.sum(dim=0) > 0  # shape: (n x n)

    num_valid_entries = valid_mask.sum(dim=1, keepdim=True)  # shape: (n x 1)

    total_similarity = (sim * mask_v.unsqueeze(0)).sum(dim=0)  # shape: (n x n)

    avg_similarity = total_similarity / num_valid_entries

    completed_sim = torch.where(valid_mask.unsqueeze(0), sim, avg_similarity.unsqueeze(0))

    return completed_sim


def threshold_tensor(tensor, threshold):
    return torch.where(tensor >= threshold, torch.ones_like(tensor), torch.zeros_like(tensor))


def calculate_label_similarity(inc_labels, inc_L_ind):
    adjusted_inc_labels = inc_labels * inc_L_ind

    joint_occurrence = torch.matmul(adjusted_inc_labels.T, adjusted_inc_labels)

    label_sums = torch.sum(adjusted_inc_labels, dim=0, keepdim=True)

    avg_label_sums = (label_sums + label_sums.T) / 2 + 1e-8

    label_similarity = joint_occurrence / avg_label_sums

    label_similarity.fill_diagonal_(0)

    return label_similarity


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        
    def wmse_loss(self,input, target, weight, reduction='mean'):
        ret = (torch.diag(weight).mm(target - input)) ** 2
        ret = torch.mean(ret)
        return ret

    
    def construct_loss(self,x_tran,x_generate,mask):
        loss = 0
        for view in range(mask.shape[1]):
            mask_view = mask[:,view].unsqueeze(1)
            mask_view_debug = mask_view.detach().cpu().numpy()
            difference = (x_tran[:,view,:] - x_generate[:,view,:]) ** 2
            loss = loss + torch.sum(difference * mask_view) / torch.sum(mask_view)
        loss = loss / mask.shape[1]
        return loss
    
    def contrastive_loss3(self, inc_labels, cls_token, inc_L_ind):
        # Compute the target similarity matrix based on inc_labels
        # Step 1: Adjust inc_labels for Missing Labels
        label_similarity = calculate_label_similarity(inc_labels, inc_L_ind)

        # Compute the predicted similarity matrix from cls_token
        # Normalize cls_token
        cls_token_normalized = F.normalize(cls_token, p=2, dim=-1)

        # If cls_token represents a single batch, use torch.mm for matrix multiplication
        if cls_token_normalized.dim() == 2:
            embedding_similarity = (1 + torch.mm(cls_token_normalized, cls_token_normalized.T)) / 2
        # If cls_token represents multiple batches, use torch.bmm for batch matrix multiplication
        elif cls_token_normalized.dim() == 3:
            embedding_similarity = (1 + torch.bmm(cls_token_normalized, cls_token_normalized.transpose(1, 2))) / 2
            # Average over the batch dimension to get a single similarity matrix
            embedding_similarity = torch.mean(embedding_similarity, dim=0)
        else:
            raise ValueError("Invalid dimension for cls_token.")

        # Calculate the loss
        loss = torch.mean((embedding_similarity - label_similarity) ** 2)
        
        #loss = self.weighted_BCE_loss(predict_similarity,label_similarity,inc_L_ind,reduction='mean')

        return loss
            
    def construct_loss2(self, reconstructed_data, original_data, mask, reduction='mean'):
        # Use sigmoid to ensure the reconstructed data is in [0, 1]
        reconstructed_data_sigmoid = torch.sigmoid(reconstructed_data)
        
        # Expand the mask to match the data dimensions
        mask_expanded = mask.unsqueeze(-1).expand_as(reconstructed_data)
        
        # Call the weighted_BCE_loss function
        loss = self.weighted_BCE_loss(reconstructed_data_sigmoid, original_data, mask_expanded, reduction)
        
        return loss
        
    
    def contrastive_loss2(self, x, inc_labels, inc_V_ind, inc_L_ind, predict,
                          epoch):  # inc_V_ind:missing views的指标矩阵，inc_L_ind:missing label的指标矩阵
        n = x.size(0)  # number of samples
        v = x.size(1)  # number of views
        if n == 1:
            return 0
        valid_labels_sum = torch.matmul(inc_L_ind.float(),
                                        inc_L_ind.float().T)  # [n, n] (nxc) * (cxn), 表示Xi与Xj之间拥有非missing标签的个数

        

        label_sum = valid_labels_sum.fill_(260.0)

        zero_ratio = torch.sum(valid_labels_sum == 0).item() / valid_labels_sum.numel()
        print(zero_ratio)

        labels1 = (torch.matmul(inc_labels, inc_labels.T) / (valid_labels_sum + 1e-9)).fill_diagonal_(
            0)
        # labels = torch.softmax(labels.masked_fill(labels==0,-1e9), dim=-1)
        # predict = threshold_tensor(predict, 0.5)
        # labels2 = (torch.matmul(predict.float(), predict.T.float()) / (label_sum + 1e-9)).fill_diagonal_(0)
        labels2 = (torch.matmul(inc_labels, inc_labels.T) / (label_sum + 1e-9)).fill_diagonal_(0)
        # labels2 = Similarity(predict)
        labels = labels1
        # labels = labels1 if Similarity_entropy(labels1) >= Similarity_entropy(labels2) else labels2  #计算T1和T2的熵，并赋值信息量大的T
        # labels = labels1

        x = F.normalize(x, p=2, dim=-1)  # [n,v,d]
        x = x.transpose(0, 1)  # [v,n,d]
        x_T = torch.transpose(x, -1, -2)  # [v,d,n]
        sim = (1 + torch.matmul(x, x_T)) / 2  # [v, n, n] #基于cosine embedding space计算的sample相似度
        sim0 = sim[0,:,:]
        mask_v = (inc_V_ind.T).unsqueeze(-1).mul((inc_V_ind.T).unsqueeze(1))  # [v, n, n]
        mask_v = mask_v.masked_fill(torch.eye(n, device=x.device) == 1, 0.)
        loss = 0
        for i in range(inc_V_ind.shape[1]):
            for j in range(i+1, inc_V_ind.shape[1]):
                inc_V_ind_sub = torch.ones
        # sim = sim * mask_v
        # sim = complete_similarity_matrix(sim,mask_v)
        # mask_v = torch.ones_like(mask_v)
        assert torch.sum(torch.isnan(mask_v)).item() == 0
        assert torch.sum(torch.isnan(labels)).item() == 0
        # assert torch.sum(torch.isnan(sim)).item() == 0
        # print('labels',torch.sum(torch.max(labels)))
        # loss = ((sim.view(v,-1)-labels.view(1,n*n))**2).mul(mask_v.view(v,-1)) # sim labels view [v, n* n]
        # try_label1 = labels1.view(1,n*n).expand(v,-1)
        # try_label2 = labels2.view(1,n*n)

        loss = self.weighted_BCE_loss(sim.view(v, -1), labels.view(1, n * n).expand(v, -1), mask_v.view(v, -1),
                                      reduction='none')
        # loss2 = self.weighted_BCE_loss(labels1.view(1,n*n).expand(v,-1),labels2.view(1,n*n).expand(v,-1), mask_v.view(v,-1), reduction='none')
        # if (epoch == 0):.
        #    loss2 = torch.zeros(loss1.shape)
        # else:
        #   loss2 = self.weighted_BCE_loss(sim.view(v,-1),labels2.view(1,n*n).expand(v,-1),mask_v.view(v,-1),reduction='none')
        # loss = loss1 + loss2.cuda()
        # assert torch.sum(torch.isnan(loss)).item() == 0
        # loss = loss1 + loss2
        loss = loss.sum(dim=-1) / (mask_v.view(v, -1).sum(dim=-1))
        return 0.5 * loss.sum() / v, sim, labels1

    def similarity_loss(self,similarity_matrices):
        num_views = len(similarity_matrices)
        l2_loss = torch.nn.MSELoss()

        total_loss = 0
        num_pairs = 0
        for i in range(num_views):
            for j in range(i + 1, num_views):
                lossi,lossj = similarity_matrices[i], similarity_matrices[j]
                loss_ij = l2_loss(similarity_matrices[i], similarity_matrices[j])
                total_loss += loss_ij
                num_pairs += 1

        # 计算均值
        mean_loss = total_loss   # 除以(m(m-1)/2)得到均值

        return mean_loss

    def calculate_masked_mse(x_origin, x_tran, mask):
        x_origin_flat = x_origin.view(-1, x_origin.size(-1))
        x_tran_flat = x_tran.view(-1, x_tran.size(-1))


        mask_flat = mask.view(-1)

        mse = torch.mean(((x_origin_flat - x_tran_flat) ** 2) * mask_flat.unsqueeze(1))
        return mse.item()


    def weighted_BCE_loss(self, target_pre, sub_target, inc_L_ind, reduction='mean'):
        assert torch.sum(torch.isnan(torch.log(target_pre))).item() == 0
        assert torch.sum(torch.isnan(torch.log(1 - target_pre + 1e-5))).item() == 0
        if (torch.sum(inc_L_ind) == 0):
            return 0
        
        target_pre_debug = target_pre[inc_L_ind==1].detach().cpu().numpy()
        sub_target_debug = sub_target[inc_L_ind==1].detach().cpu().numpy()
        
        
        res = torch.abs((sub_target.mul(torch.log(target_pre + 1e-5)) \
                         + (1 - sub_target).mul(torch.log(1 - target_pre + 1e-5)))).mul(inc_L_ind)
        
            
        res_debug = res.detach().cpu().numpy()

        if reduction == 'mean':
            return torch.sum(res) / torch.sum(inc_L_ind)
        elif reduction == 'sum':
            return torch.sum(res)
        elif reduction == 'none':
            return res

    def BCE_loss(self, target_pre, sub_target):
        return torch.mean(torch.abs((sub_target.mul(torch.log(target_pre + 1e-10)) \
                                     + (1 - sub_target).mul(torch.log(1 - target_pre + 1e-10)))))

    def weighted_BCE_loss2(self, target_pre, sub_target, inc_L_ind, reduction='mean'):
        assert torch.sum(torch.isnan(torch.log(target_pre))).item() == 0
        assert torch.sum(torch.isnan(torch.log(1 - target_pre + 1e-5))).item() == 0
        res = torch.abs((sub_target.mul(torch.log(target_pre + 1e-5)) \
                         + (1 - sub_target).mul(torch.log(1 - target_pre + 1e-5))))

        if reduction == 'mean':
            return torch.sum(res) / torch.sum(torch.ones_like(inc_L_ind))
        elif reduction == 'sum':
            return torch.sum(res)
        elif reduction == 'none':
            return res

    def L2_norm_loss(Y, pseduo_label):
        loss = Y - pseduo_label
        return torch.norm(loss, p=2)


class SPLC(nn.Module):
    def __init__(self,
                 tau: float = 0.6,
                 change_epoch: int = 1,
                 margin: float = 1.0,
                 gamma: float = 2.0,
                 reduction: str = 'sum') -> None:
        super(SPLC, self).__init__()
        self.tau = tau
        self.change_epoch = change_epoch
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch) -> torch.Tensor:

        # Subtract margin for positive logits
        logits = torch.where(targets == 1, logits - self.margin, logits)

        # SPLC missing label corection
        if epoch >= self.change_epoch:
            targets = torch.where(
                torch.sigmoid(logits) > self.tau,
                torch.tensor(1).cuda(), targets)

        pred = torch.sigmoid(logits)

        # Focal margin for positive loss
        pt = (1 - pred) * targets + pred * (1 - targets)
        focal_weight = pt ** self.gamma

        los_pos = targets * F.logsigmoid(logits)
        los_neg = (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss