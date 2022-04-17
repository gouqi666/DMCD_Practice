import xgboost as xgb
import pickle
from datasets import TestDataset
import time
import yaml
import argparse
import pandas as pd
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ConfigPath',default = "./config/config.yml")
    args = parser.parse_args()
    fs = open(args.ConfigPath, 'r', encoding="UTF-8")
    baseConfig = yaml.load(fs, Loader=yaml.FullLoader)
    start = time.time()
    print("正在读取数据--------------")
    test_dataset = TestDataset(baseConfig)
    print("数据读取完成：%.5f---",time.time() - start)


    # load xgb  from file
    loaded_model = pickle.load(open(baseConfig['FilePath']['save_models_xgb'], "rb"))
    x_test = xgb.DMatrix(test_dataset.data)
    y_pred = loaded_model.predict(x_test)
    predictions = [round(value) for value in y_pred]
    ret = pd.DataFrame(columns=['Id','Expected'])
    ret['Id'] = [i for i in range(len(test_dataset))]
    ret['Expected'] = predictions
    ret.to_csv(baseConfig['FilePath']['outputFiles_xgb'],index=None)

    # load random_forest model from file
    loaded_model = pickle.load(open(baseConfig['FilePath']['save_models_ran'], "rb"))
    y_pred = loaded_model.predict(test_dataset.data)
    predictions = [round(value) for value in y_pred]
    ret = pd.DataFrame(columns=['Id','Expected'])
    ret['Id'] = [i for i in range(len(test_dataset))]
    ret['Expected'] = predictions
    ret.to_csv(baseConfig['FilePath']['outputFiles_ran'],index=None)



