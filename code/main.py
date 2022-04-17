import torch.utils.data
import yaml
import json
import os
# from config.BaseConfig import BaseConfig
import time
import matplotlib.pyplot as plt
import argparse
from datasets import TrainDataset
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from test import test_dataset
from entity import Author,Conference,Journal,Paper
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ConfigPath',default = "./config/config.yml")
    args = parser.parse_args()
    # baseConfig = BaseConfig(args.ConfigPath)
    fs = open(args.ConfigPath, 'r', encoding="UTF-8")
    baseConfig = yaml.load(fs, Loader=yaml.FullLoader)
    start = time.time()
    print("正在读取数据--------------")
    # train_dataset = TrainDataset(baseConfig)
    print("数据读取完成：%.5f---",time.time() - start)
    # train_dataset,valid_dataset = torch.utils.data.random_split(train_dataset,[])

    # for batch in train_dataset.data:
    #     print(batch)
    #     break
    test_dataset = test_dataset()
    x_train, x_valid, y_train, y_valid = train_test_split(test_dataset.data, test_dataset.label, test_size=0.1)
    train = xgb.DMatrix(x_train, label=y_train)
    valid = xgb.DMatrix(x_valid, label=y_valid)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 2,
        'max_depth': 6,
        'eta': 0.1,
        'silent': 1,
        'gamma': 0,
        'min_child_weight': 1,
        'subsample': 1,
        'colsample_bytree': 1,
        'colsample_bylevel': 1,
        'lambda': 1,
        'alpha': 0,
        'nthread': -1,
        'eval_metric': 'merror',
        'seed': 0
    }
    evallist = [(valid, 'eval')]
    model = xgb.train(list(params.items()), train, 5, evals=evallist)
    pickle.dump(model, open("pima.pickle.dat", "wb"))
    xgb.plot_importance(model)
    plt.show()
    # load model from file


    loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
    x_test = xgb.DMatrix([[1],[2],[3],[124354],[533424],[689784]])
    y_pred = loaded_model.predict(x_test)
    predictions = [round(value) for value in y_pred]
    print(y_pred)
    #
    # for author,conference,journal,paper in zip(Author,Conference,Journal,Paper):
    #     print(author)
    #     print(conference)
    #     print(journal)
    #     print(paper)
    #     break



