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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from entity import Author,Conference,Journal,Paper

class XgbTrainer():
    def __init__(self,train_dataset,baseConfig):
        self.baseConfig = baseConfig
        # x_train,y_train = train_dataset.data, train_dataset.label
        # self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(train_dataset.data, train_dataset.label, test_size=0.1)
        self.x_train,self.y_train = train_dataset.data, train_dataset.label
        self.train_data = xgb.DMatrix(self.x_train, label=self.y_train)

        # self.valid_data = xgb.DMatrix(self.x_valid, label=self.y_valid)
        self.params = {
            'booster': 'gbtree',
            'objective': 'multi:softmax',
            'num_class': 2,
            'max_depth': 10,
            'eta': 0.1,
            'silent': 1,
            'gamma': 0.1,
            'min_child_weight': 2,
            'subsample': 1,
            'colsample_bytree': 1,
            'colsample_bylevel': 1,
            'lambda': 1,
            'alpha': 0,
            'nthread': -1,
            'eval_metric': 'merror',
            'seed': 40
        }
        # self.evallist = [(self.valid_data, 'eval')]

    def train(self):
        print("开始训练----------")
        start_train = time.time()
        model = xgb.train(list(self.params.items()), self.train_data, 10)  #, evals=self.evallist
        pickle.dump(model, open(self.baseConfig['FilePath']['save_models_xgb'], "wb"))
        xgb.plot_importance(model,importance_type='gain')
        plt.show()
        print("训练结束：%.5f------", time.time() - start_train)

class RandomForestTrainer():
    def __init__(self,train_dataset,baseConfig):
        self.baseConfig = baseConfig
        self.x_train,self.y_train = train_dataset.data, train_dataset.label
        # self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(train_dataset.data, train_dataset.label, test_size=0.1)
        self.clf = RandomForestClassifier(n_estimators= 200,max_depth=10, random_state=47)
    def train(self):
        print('random_forest-------')
        self.clf.fit(self.x_train,self.y_train)
        print(self.clf.feature_importances_)
        # pred = self.clf.predict(self.x_valid)
        # accur = 0
        # for p,l in zip(pred,self.y_valid):
        #     if p == l:
        #         accur += 1
        # print(accur / len(self.y_valid))
        pickle.dump(self.clf, open(self.baseConfig['FilePath']['save_models_ran'], "wb"))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ConfigPath',default = "./config/config.yml")
    args = parser.parse_args()
    # baseConfig = BaseConfig(args.ConfigPath)
    fs = open(args.ConfigPath, 'r', encoding="UTF-8")
    baseConfig = yaml.load(fs, Loader=yaml.FullLoader)
    start = time.time()
    print("正在读取数据--------------")
    train_dataset = TrainDataset(baseConfig)
    print("数据读取完成：%.5f---",time.time() - start)
    xgb_trainer = XgbTrainer(train_dataset,baseConfig)
    xgb_trainer.train()
    random_forest = RandomForestTrainer(train_dataset,baseConfig)
    random_forest.train()