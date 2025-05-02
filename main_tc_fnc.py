from fuction_utility import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from args import *
from models.mutual_learning import CFML
from tool.utils import set_logger,backup_codes
import logging

def main(k,Config):
    # ten-fold cross validation
    folds, X_train, y_train = load_data_kfold(k = 10,path = Config.data_path,random_state=Config.random_state)
    folds, X_train_tc, y_train_tc = load_tc(k=10, path=Config.tc_path, random_state=Config.random_state)
    y = np.array([1]).reshape(-1, 1)
    prediction = np.array([1]).reshape(-1, 1)
    y_score = np.array([1]).reshape(-1, 1)

    for i, (train_idx, val_idx) in enumerate(folds):
        print('Fold', i + 1)
        # prepare data
        X_train_cv = X_train[train_idx]
        y_train_cv = y_train[train_idx]
        X_test_cv = X_train[val_idx]
        y_test_cv = y_train[val_idx]
        X_train_cv_tc = X_train_tc[train_idx]
        X_test_cv_tc = X_train_tc[val_idx]

        # training model
        pro, y_submission, model = CFML(Config, X_train_cv, y_train_cv, X_test_cv, X_train_cv_tc,X_test_cv_tc,i=i, k=k)

        prediction = np.hstack((prediction, y_submission.reshape(1, -1)))
        y_score = np.hstack((y_score, pro.reshape(1, -1)))
        y = np.hstack((y, y_test_cv.T))
    prediction = prediction[:, 1:]
    y = y[:, 1:]
    y_score = y_score[:,1:]

    # compute evaluation metrics

    acc, spe, sen, f1 ,roc_auc,fpr,tpr = acc_pre_recall_f(y.T, prediction.T,y_score.T)
    print(acc,spe,sen,f1,roc_auc)
    logging.info(acc)
    logging.info(spe)
    logging.info(sen)
    logging.info(f1)
    logging.info(roc_auc)
    return acc,spe,sen,f1,roc_auc

def count_variance(Config):
    n = 1
    results = np.zeros((5, n))
    mean_variance = np.zeros((5, 3))
    for i in range(n):

        acc, spe, sen, f1, roc_auc = main(i,Config)
        results[0, i] = acc
        results[1, i] = spe
        results[2, i] = sen
        results[3, i] = f1
        results[4, i] = roc_auc
    print(results)
    for i in range(5):
        mean_variance[i, 0] = np.mean(results[i, :])
        mean_variance[i, 1] = np.var(results[i, :])
        mean_variance[i, 2] = np.std(results[i, :])
    output_path = os.path.join(Config.result_path,Config.alg+'_output.mat')
    savemat(output_path,mdict={'results':results,'mean_variance_std': mean_variance})
    print (mean_variance)

if __name__ == '__main__':
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Config Setting
    Config = args
    Config.workspace = os.path.join('workspace', 'fusion',now)
    Config.model_path = os.path.join(Config.workspace, 'models')
    Config.result_path = os.path.join(Config.workspace, 'result')
    os.makedirs(Config.model_path, exist_ok=True)
    os.makedirs(Config.result_path, exist_ok=True)

    # save code for backups
    set_logger(Config.workspace, 'logs.txt')
    backup_root = os.path.join(Config.workspace, 'reproducibility')
    backup_codes(backup_root)

    # training and evaluation
    count_variance(Config)
