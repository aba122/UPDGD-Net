
# this code is released by: 
# L. Yang, X.-Z. Wu, Y. Jiang, and Z.-H. Zhou. 
# Multi-label deep forest. 
#In: Proceedings of the 24th European Conference on Artificial Intelligence (ECAI'20), 
# Santiago de Compostela, Spain, 2020. [code]

import numpy as np
from sklearn import metrics

def do_metric(y_prob, label):

    y_predict = y_prob > 0.5
    ranking_loss = 1 - compute_ranking_loss(y_prob, label)
    # print(ranking_loss)
    one_error = compute_one_error(y_prob, label)
    # print(one_error)
    coverage = compute_coverage(y_prob, label)
    # print(coverage)
    hamming_loss = 1 - compute_hamming_loss(y_predict, label)
    # print(hamming_loss)
    precision = compute_average_precision(y_prob, label)
    # print(precision)
    macro_f1 = compute_macro_f1(y_predict, label)
    # print(macro_f1)
    micro_f1 = compute_micro_f1(y_predict, label)
    # print(micro_f1)
    auc = compute_auc(y_prob, label)
    auc_me = mlc_auc(y_prob, label)
    return np.array([precision, hamming_loss, ranking_loss, auc_me, one_error, coverage, auc, macro_f1, micro_f1])
    
def init_supervise(supervise):
    if supervise == "ranking loss":
        ranking_loss = 1.0
        return ranking_loss
    elif supervise == "hamming loss":
        hamming_loss = 1.0
        return hamming_loss
    elif supervise == "one-error":
        one_error = 1.0
        return one_error
    elif supervise == "average precision":
        average_precision = 0.0
        return average_precision
    elif supervise == "micro-f1":
        micro_f1 = 0.0
        return micro_f1
    elif supervise == "macro-f1":
        macro_f1 = 0.0
        return macro_f1
    elif supervise == "coverage":
        coverage = 1000.0
        return coverage
    elif supervise == "macro_auc":
        macro_auc = 0.0
        return macro_auc


def compare_supervise_value(supervise, supervise_value1, supervise_value2):
    if supervise == "ranking loss" or supervise == "hamming loss" or supervise == "one-error" or supervise == "coverage":
        if supervise_value1 > supervise_value2 + 1e-4:
            return False
        else:
            return True
    elif supervise == "average precision" or supervise == "micro-f1" or supervise == "macro-f1" or supervise == "macro_auc":
        if supervise_value1 + 1e-4 < supervise_value2:
            return False
        else:
            return True


def compute_supervise(supervise, y_prob, label, threshold):
    predict = y_prob > threshold
    if supervise == "ranking loss":
        temp_ranking_loss = compute_ranking_loss(
            y_prob, label)  # prob / y_prob
        value = temp_ranking_loss
    elif supervise == "hamming loss":
        temp_hamming_loss = compute_hamming_loss(predict, label)
        value = temp_hamming_loss
    elif supervise == "one-error":
        temp_one_error = compute_one_error(y_prob, label)
        value = temp_one_error
    elif supervise == "average precision":
        temp_average_precision = compute_average_precision(y_prob, label)
        value = temp_average_precision
    elif supervise == "micro-f1":
        temp_micro_f1 = compute_micro_f1(predict, label)
        value = temp_micro_f1
    elif supervise == "macro-f1":
        temp_macro_f1 = compute_macro_f1(predict, label)
        value = temp_macro_f1
    elif supervise == "coverage":
        temp_coverage = compute_coverage(y_prob, label)
        value = temp_coverage
    elif supervise == "macro_auc":
        macro_auc = compute_auc(y_prob, label)
        value = macro_auc
    return value



def compute_supervise_vec(supervise, y_prob, label, threshold):
    predict = y_prob > threshold
    if supervise == "ranking loss":
        temp_ranking_loss = compute_ranking_loss_vec(
            y_prob, label)  # prob / y_prob
        value = temp_ranking_loss
    elif supervise == "hamming loss":
        temp_hamming_loss = compute_hamming_loss_vec(predict, label)
        value = temp_hamming_loss
    elif supervise == "one-error":
        temp_one_error = compute_one_error_vec(y_prob, label)
        value = temp_one_error
    elif supervise == "average precision":
        temp_average_precision = compute_average_precision_vec(y_prob, label)
        value = temp_average_precision
    elif supervise == "coverage":
        temp_coverage = compute_coverage_vec(y_prob, label)
        value = temp_coverage
    elif supervise == "macro_auc":
        macro_auc = compute_auc_vec(y_prob, label)
        value = macro_auc
    return value


def update_supervise(supervise, value_pool, layer_index, y_prob, label, threshold):
    back = False
    back2 = False
    value_pool[layer_index] = compute_supervise(
        supervise, y_prob, label, threshold)
    if layer_index >= 2 and compare_supervise_value(supervise, value_pool[layer_index - 2],
                                                    value_pool[layer_index - 1]):
        back2 = True
    if layer_index >= 1 and compare_supervise_value(supervise, value_pool[layer_index - 1], value_pool[layer_index]):
        back = True
    return [back, back2]


def compute_accuracy(pred_label, label):
    num_samples = len(label)
    acc = sum(label == pred_label) * 1.0 / num_samples
    return acc


def compute_performance_single_label(predict_score, label):
    predict_label = predict_score > 0.5
    _, num_labels = label.shape
    acc = np.empty(num_labels)
    f1 = np.empty(num_labels)
    auc = np.empty(num_labels)
    for i in range(num_labels):
        acc[i] = metrics.accuracy_score(
            label[:, i].reshape(-1), predict_label[:, i].reshape(-1))
        f1[i] = metrics.f1_score(
            label[:, i].reshape(-1), predict_label[:, i].reshape(-1))
        auc[i] = metrics.roc_auc_score(
            label[:, i].reshape(-1), predict_score[:, i].reshape(-1))
    return [acc, f1, auc]


def compute_rank(y_prob):
    rank = np.zeros(y_prob.shape)
    for i in range(len(y_prob)):
        temp = y_prob[i, :].argsort() #argsort()返回数组值从小到大的索引
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(y_prob[i, :]))
        rank[i, :] = ranks
    return y_prob.shape[1] - rank


# example based measure
def compute_hamming_loss(pred_label, label):
    acc = compute_accuracy(pred_label, label)
    return 1 - acc.mean()


def compute_hamming_loss_vec(pred_label, label):
    acc = compute_accuracy(pred_label, label)
    return 1 - acc


# label based measure
def compute_macro_f1(pred_label, label):
    up = np.sum(pred_label * label, axis=0)
    down = np.sum(pred_label, axis=0) + np.sum(label, axis=0)
    if np.sum(np.sum(label, axis=0) == 0) > 0:
        up[down == 0] = 0
        down[down == 0] = 1
    macro_f1 = 2.0 * np.sum(up / down)
    macro_f1 = macro_f1 * 1.0 / label.shape[1]
    return macro_f1


def compute_micro_f1(pred_label, label):
    up = np.sum(pred_label * label)
    down = np.sum(pred_label) + np.sum(label)
    if np.sum(np.sum(label) == 0) > 0:
        up[down == 0] = 0
        down[down == 0] = 1
    micro_f1 = 2.0 * up / down
    return micro_f1


# ranking based measure
def compute_ranking_loss(y_prob, label):
    # y_predict = y_prob > 0.5
    num_samples, num_labels = label.shape
    loss = 0
    for i in range(num_samples):
        prob_positive = y_prob[i, label[i, :] > 0.5]
        prob_negative = y_prob[i, label[i, :] < 0.5]
        s = 0
        for j in range(prob_positive.shape[0]):
            for k in range(prob_negative.shape[0]):
                if prob_negative[k] >= prob_positive[j]:
                    s += 1

        label_positive = np.sum(label[i, :] > 0.5)
        label_negative = np.sum(label[i, :] < 0.5)
        if label_negative != 0 and label_positive != 0:
            loss = loss + s * 1.0 / (label_negative * label_positive)

    return loss * 1.0 / num_samples


def compute_ranking_loss_vec(y_prob, label):
    num_samples, num_labels = label.shape
    loss = np.zeros(num_samples)
    for i in range(num_samples):
        prob_positive = y_prob[i, label[i, :] > 0.5]
        prob_negative = y_prob[i, label[i, :] < 0.5]
        s = 0
        for j in range(prob_positive.shape[0]):
            for k in range(prob_negative.shape[0]):
                if prob_negative[k] >= prob_positive[j]:
                    s += 1

        label_positive = np.sum(label[i, :] > 0.5)
        label_negative = np.sum(label[i, :] < 0.5)
        if label_negative != 0 and label_positive != 0:
            loss[i] = s * 1.0 / (label_negative * label_positive)
    return loss


def compute_one_error(y_prob, label):
    num_samples, num_labels = label.shape
    loss = 0
    for i in range(num_samples):
        pos = np.argmax(y_prob[i, :])
        loss += label[i, pos] < 0.5
    return loss * 1.0 / num_samples


def compute_one_error_vec(y_prob, label):
    num_samples, num_labels = label.shape
    loss = np.zeros(num_samples)
    for i in range(num_samples):
        pos = np.argmax(y_prob[i, :])
        loss[i] = label[i, pos] < 0.5
    return loss


def compute_coverage(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    coverage = 0
    for i in range(num_samples):
        if sum(label[i, :] > 0.5) > 0:
            coverage += max(rank[i, label[i, :] > 0.5])
    coverage = coverage * 1.0 / num_samples - 1
    return coverage / num_labels


def compute_coverage_vec(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    coverage = np.zeros(num_samples)
    for i in range(num_samples):
        if sum(label[i, :] > 0.5) > 0:
            coverage[i] = max(rank[i, label[i, :] > 0.5])
    return coverage


def compute_average_precision(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob) #每个sample的标签预测概率排序
    precision = 0
    for i in range(num_samples):
        positive = np.sum(label[i, :] > 0.5)
        rank_i = rank[i, label[i, :] > 0.5] #找出第i个sample中的的正标签在预测值中的排名
        temp = rank_i.argsort() #argsort()返回数组值从小到大的索引排名
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(rank_i))
        ranks = ranks + 1
        ans = ranks * 1.0 / rank_i
        if positive > 0:
            precision += np.sum(ans) * 1.0 / positive
    return precision / num_samples


def compute_average_precision_vec(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    precision = np.zeros(num_samples)
    for i in range(num_samples):
        positive = np.sum(label[i, :] > 0.5)
        rank_i = rank[i, label[i, :] > 0.5]
        temp = rank_i.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(rank_i))
        ranks = ranks + 1
        ans = ranks * 1.0 / rank_i
        if positive > 0:
            precision[i] = np.sum(ans) * 1.0 / positive
    return precision


def compute_auc(y_prob, label):
    n, m = label.shape
    macro_auc = 0
    valid_labels = 0
    for i in range(m):
        if np.unique(label[:, i]).shape[0] == 2:
            index = np.argsort(y_prob[:, i])
            pred = y_prob[:, i][index]
            y = label[:, i][index] + 1
            fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
            temp = metrics.auc(fpr, tpr)
            macro_auc += temp
            valid_labels += 1
    macro_auc /= valid_labels
    return macro_auc

def compute_mlr_auc(y_prob, label):
    n, m = label.shape
    macro_auc = 0
    valid_labels = 0
    fpr = np.zeros(m)
    tpr = np.zeros(m)
    for i in range(m):
        if np.unique(label[:, i]).shape[0] == 2:
            index = np.argsort(y_prob[:, i])
            pred = y_prob[:, i][index]
            y = label[:, i][index] + 1
            fpr[i], tpr[i], thresholds = metrics.roc_curve(y, pred, pos_label=2)
    area = 0
    for i in range(m):
        area = area+(fpr[i+1]-fpr[i])*(tpr[i+1]+tpr[i])*0.5
    mlr_auc = area/(fpr[m]-fpr[1])
    return mlr_auc

def compute_auc_vec(y_prob, label):
    n, m = label.shape
    macro_auc = np.zeros(m)
    valid_labels = 0
    for i in range(m):
        if np.unique(label[:, i]).shape[0] == 2:
            index = np.argsort(y_prob[:, i])
            pred = y_prob[:, i][index]
            y = label[:, i][index] + 1
            fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
            temp = metrics.auc(fpr, tpr)
            macro_auc[i] = temp
            valid_labels += 1
    return macro_auc

def performance(y,f,T):
#    code is written by Jerry, according to the original code from 
#    from http://mlda.swu.edu.cn/codes.php?name=iMVWL
    n,K = f.shape    
    match = np.zeros(n)
    fn = np.zeros(n)
    fp = np.zeros(n)
    for i in range(n):
        si = f[i,:].argsort()[::-1]        
        words=y[i,:]
        correct_labels=np.where(words>-1)
        correct_labels = (np.array(correct_labels)).reshape(-1)
        si = si[0:T]   # T numbers
        match[i] = 0
        for j in range(len(correct_labels)):
            if np.where(si==correct_labels[j])[0].shape[0]!=0:
                match[i] = match[i]+1
        fn[i] = len(correct_labels)-match[i]
        fp[i] = T-match[i]
    return match,fp,fn
  
def mlr_roc(f, y_test):
#    code is written by Jerry, according to the original code from 
#    from http://mlda.swu.edu.cn/codes.php?name=iMVWL
    K = y_test.shape[1]
    tpr1 = np.zeros(K)
    fpr1 = np.zeros(K)
    
    for i in range(K):
        match,fpp,fnn = performance(y_test,f,i+1);
        tp1=match.sum()
        fn1=fnn.sum()
        fp1=fpp.sum()
        tn1 = K*f.shape[0]-(tp1+fp1+fn1)
        tpr1[i] = tp1/(tp1+fn1)
        fpr1[i] = fp1/(fp1+tn1)
    return tpr1,fpr1  
def mlc_auc(rocZ,newY):
#    code is written by Jerry, according to the original code from 
#    from http://mlda.swu.edu.cn/codes.php?name=iMVWL    
#    rocZ: problistic matrix  n*c
#    newY: n*c matrix,elements in {-1,1}
    if newY.min()==0:
        newY = newY*2-1
    
    tpr,fpr = mlr_roc(rocZ,newY)
    area = 0
    m = newY.shape[1]
    for i in range(m-1):
        area = area+(fpr[i+1]-fpr[i])*(tpr[i+1]+tpr[i])*0.5
    value_auc = area/(fpr[m-1]-fpr[0])
    return value_auc

if __name__=='__main__':
    a=np.array([[0.1,0.2,0.1,0.4,0.8,0.9],[0.1,0.2,0.1,0.4,0.8,0.9]])
    b=np.array([[1.,0.,1.,0.,1.,1.],[0.,1.,0.,1.,0.,0.]])
    print(do_metric(a,b))
