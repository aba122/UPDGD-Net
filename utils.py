import numpy as np
import logging
import os
import time
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0

    def reset(self):
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.vals.append(val)
        self.sum = np.sum(self.vals)
        self.count = len(self.vals)
        self.avg = np.mean(self.vals)
        self.std = np.std(self.vals)
        self.min = min(self.vals)
        self.min_ind = self.vals.index(self.min)
        self.max = max(self.vals)
        self.max_ind = self.vals.index(self.max)

def setLogger(logfile):
    logger = logging.getLogger()
    
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console = logging.StreamHandler()
    
    
    while logger.handlers:
        logger.handlers.pop()
    if logfile:
        handler = logging.FileHandler(logfile,mode='w') 
        logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def save_epoch_results(file_path, epoch, batch_time, data_time, losses, evaluation_results, args):
    with open(file_path, mode='a') as file.handle:
        if os.path.getsize(file_path) == 0:
            file_handle.write('Epoch Time Data Loss AP HL RL AUC lr alpha beta gamma\n')
            
        res_list = [
            epoch,
            f"{batch_time.avg:.3f}",
            f"{data_time.avg:.3f}"
            f"{losses.avg:.3f}",
            f"{evaluation_results[0]:.3f}",
            f"{evaluation_results[1]:.3f}",
            f"{evaluation_results[2]:.3f}",
            f"{evaluation_results[3]:.3f}",
            args.lr,
            args.alpha,
            args.beta,
            args.gama
        ]
        
        res_str = ''.join(map(str, res_list))
        file_handle.write(res_str)
        file_handle.write('\n')
        