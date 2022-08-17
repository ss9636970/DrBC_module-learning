import logging
import numpy as np

def create_logger(path, log_file):
    # config
    logging.captureWarnings(True)     # 捕捉 py waring message
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    my_logger = logging.getLogger(log_file) #捕捉 py waring message
    my_logger.setLevel(logging.INFO)
    
    # file handler
    fileHandler = logging.FileHandler(path + log_file, 'w', 'utf-8')
    fileHandler.setFormatter(formatter)
    my_logger.addHandler(fileHandler)
    
    # console handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    my_logger.addHandler(consoleHandler)
    
    return my_logger

#logger.disabled = True  #暫停 logger
#logger.handlers  # logger 內的紀錄程序
#logger.removeHandler  # 移除紀錄程序
#logger.info('xxx', exc_info=True)  # 紀錄堆疊資訊

# top  n  %  accuracy
# both preds and truths are same shape m by n (m is number of predictions and n is number of classes)
def top_n_accuracy(preds, truths, n):
    best_n = np.argsort(preds)[-n:]
    ts = np.argsort(truths)[-n:]
    successes = 0
    for i in range(ts.shape[0]):
      if best_n[i] in ts:
        successes += 1
    return float(successes / n)



#計算 Kendall tau distance
def normalised_kendall_tau_distance(values1, values2):
    """Compute the Kendall tau distance."""
    n = len(values1)
    if n != len(values2):
        print('wrong length')
        return 0
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    ndisordered = np.logical_or(np.logical_and(values1[i] < values1[j], values2[i] < values2[j]), np.logical_and(values1[i] > values1[j], values2[i] > values2[j])).sum()
    return ndisordered / (n * (n - 1))