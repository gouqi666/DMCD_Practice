import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils import construct_features
from entity import Entity
class TrainDataset(Dataset):
    def __init__(self,baseConfig,):
        super(TrainDataset,self).__init__()
        self.Entity = Entity(baseConfig)
        df = pd.read_csv(baseConfig['FilePath']['train'])
        self.data = []
        self.label = []
        for index, row in df.iterrows():
            AuthorId = row['AuthorId']
            for PaperId in  row['ConfirmedPaperIds'].split():
                PaperId = int(PaperId)
                feat = construct_features(AuthorId,PaperId,self.Entity)
                self.data.append(feat)
                self.label.append(1)
            for PaperId in row['DeletedPaperIds'].split():
                PaperId = int(PaperId)

                feat = construct_features(AuthorId,PaperId,self.Entity)
                self.data.append(feat)
                self.label.append(0)
    def __getitem__(self, item):
        return self.data[item],self.label[item]
    def __len__(self):
        return len(self.data)

class TestDataset(Dataset):
    def __init__(self,baseConfig):
        super(TestDataset,self).__init__()
        self.Entity = Entity(baseConfig)
        df = pd.read_csv(baseConfig['FilePath']['test2'])
        self.data = []
        for index, row in df.iterrows():
            AuthorId = row['AuthorId']
            PaperId = row['PaperId']
            feat = construct_features(AuthorId,PaperId,self.Entity)
            self.data.append(feat)
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)