import numpy as np
import pandas as pd
import math
from torch.utils.data import Dataset
from collections import defaultdict
import pickle
class Entity():
    def __init__(self,baseConfig):

        f_read = open(baseConfig['FilePath']['conference_feature'], 'rb')
        self.Conference = pickle.load(f_read)
        f_read.close()
        f_read = open(baseConfig['FilePath']['journal_feature'], 'rb')
        self.Journal = pickle.load(f_read)
        f_read.close()
        f_read = open(baseConfig['FilePath']['paper_feature'], 'rb')
        self.Paper = pickle.load(f_read)
        f_read.close()
        f_read = open(baseConfig['FilePath']['author_feature'], 'rb')
        self.Author = pickle.load(f_read)
        f_read.close()
        # self.Author = defaultdict(dict)
        # self.Conference = defaultdict(dict)
        # self.Journal = defaultdict(dict)
        # self.Paper = defaultdict(dict)
        # df = pd.read_csv(baseConfig['FilePath']['author'])
        # for index,row in df.iterrows():
        #     self.Author[row['Id']]['Name'] = row['Name']
        #     self.Author[row['Id']]['Affiliation'] = None if pd.isnull(row['Affiliation']) else row['Affiliation']
        #
        # df = pd.read_csv(baseConfig['FilePath']['conference'])
        # for index, row in df.iterrows():
        #     self.Conference[row['Id']]['ShortName'] = None if pd.isnull(row['ShortName']) else row['ShortName']
        #     self.Conference[row['Id']]['FullName'] =  None if pd.isnull(row['FullName']) else row['FullName']
        #     self.Conference[row['Id']]['HomePage'] = None if pd.isnull(row['HomePage']) else row['HomePage']
        #
        # df = pd.read_csv(baseConfig['FilePath']['journal'])
        # for index, row in df.iterrows():
        #     self.Journal[row['Id']]['ShortName'] = None if pd.isnull(row['ShortName']) else row['ShortName']
        #     self.Journal[row['Id']]['FullName'] =  None if pd.isnull(row['FullName']) else row['FullName']
        #     self.Journal[row['Id']]['HomePage'] = None if pd.isnull(row['HomePage']) else row['HomePage']
        #
        # df = pd.read_csv(baseConfig['FilePath']['paper'])
        # for index, row in df.iterrows():
        #     self.Paper[row['Id']]['Title'] = None if pd.isnull(row['Title']) else row['Title']
        #     self.Paper[row['Id']]['Year'] = None if pd.isnull(row['Year']) else row['Year']
        #     self.Paper[row['Id']]['ConferenceId'] = None if pd.isnull(row['ConferenceId']) else row['ConferenceId']
        #     self.Paper[row['Id']]['JournalId'] = None if pd.isnull(row['JournalId']) else row['JournalId']
        #     self.Paper[row['Id']]['Keyword'] = None if pd.isnull(row['Keyword']) else row['Keyword']
        #
        #
        # # 处理 PaperAuthor文件
        # df = pd.read_csv(baseConfig['FilePath']['paperAuthor'])
        # for index,row in df.iterrows():
        #     PaperId = row['PaperId']
        #     AuthorId = row['AuthorId']
        #     Name = row['Name']
        #     Affiliation = row['Affiliation']
        #     if self.Paper[PaperId].get('Author',None) == None:
        #         self.Paper[PaperId]['Author'] = []
        #     self.Paper[PaperId]['Author'].append({'Id':AuthorId,'Name':None if pd.isnull(Name) else Name,
        #                                           'Affiliation':None if pd.isnull(Affiliation) else Affiliation})
        #     if self.Author[AuthorId].get('Paper',None) == None:
        #         self.Author[AuthorId]['Paper'] = []
        #     self.Author[AuthorId]['Paper'].append(PaperId)
        #
        # f_save = open(baseConfig['FilePath']['paper_feature'], 'wb')
        # pickle.dump(self.Paper, f_save)
        # f_save.close()
        #
        # f_save = open(baseConfig['FilePath']['author_feature'], 'wb')
        # pickle.dump(self.Author, f_save)
        # f_save.close()
        #
        # f_save = open(baseConfig['FilePath']['conference_feature'], 'wb')
        # pickle.dump(self.Conference, f_save)
        # f_save.close()
        #
        # f_save = open(baseConfig['FilePath']['journal_feature'], 'wb')
        # pickle.dump(self.Journal, f_save)
        # f_save.close()
class Author(Dataset):
    def __init__(self,author_path):
        df = pd.read_csv(author_path)
        # print(df.columns.values.tolist())
        self.Id = []
        self.Name = []
        self.Affiliation = []
        for index,row in df.iterrows():
            self.Id.append(row['Id'])
            self.Name.append(row['Name'])
            self.Affiliation.append(None if pd.isnull(row['Affiliation']) else row['Affiliation'])
    def __getitem__(self, item):
        return {"Id": self.Id[item], 'Name': self.Name[item], "Affiliation" : self.Affiliation[item]}
    def __len__(self):
        return len(self.Id)

class Conference(Dataset):
    def __init__(self,conf_path):
        df = pd.read_csv(conf_path)
        # print(df.columns.values.tolist())
        self.Id = []
        self.ShortName = []
        self.FullName = []
        self.HomePage = []
        for index, row in df.iterrows():
            self.Id.append(row['Id'])
            self.ShortName.append(None if pd.isnull(row['ShortName']) else row['ShortName'])
            self.FullName.append(None if pd.isnull(row['FullName']) else row['FullName'])
            self.HomePage.append(None if pd.isnull(row['HomePage']) else row['HomePage'])
    def __getitem__(self, item):
        return {"Id": self.Id[item], 'ShortName': self.ShortName[item],"FullName": self.FullName[item],"HomePage": self.HomePage[item]}

    def __len__(self):
        return len(self.Id)

class Journal(Dataset):
    def __init__(self,conf_path):
        df = pd.read_csv(conf_path)
        # print(df.columns.values.tolist())
        self.Id = []
        self.ShortName = []
        self.FullName = []
        self.HomePage = []
        for index, row in df.iterrows():
            self.Id.append(row['Id'])
            self.ShortName.append(None if pd.isnull(row['ShortName']) else row['ShortName'])
            self.FullName.append(None if pd.isnull(row['FullName']) else row['FullName'])
            self.HomePage.append(None if pd.isnull(row['HomePage']) else row['HomePage'])
    def __getitem__(self, item):
        return {"Id": self.Id[item], 'ShortName': self.ShortName[item],"FullName": self.FullName[item],"HomePage": self.HomePage[item]}

    def __len__(self):
        return len(self.Id)

class Paper(Dataset):
    def __init__(self,conf_path):
        df = pd.read_csv(conf_path)
        # print(df.columns.values.tolist())
        self.Id = []
        self.Title = []
        self.Year = []
        self.ConferenceId = []
        self.JournalId = []
        self.Keyword = []
        for index, row in df.iterrows():
            self.Id.append(row['Id'])
            self.Title.append(None if pd.isnull(row['Title']) else row['Title'])
            self.Year.append(None if pd.isnull(row['Year']) else row['Year'])
            self.ConferenceId.append(None if pd.isnull(row['ConferenceId']) else row['ConferenceId'])
            self.JournalId.append(None if pd.isnull(row['JournalId']) else row['JournalId'])
            self.Keyword.append(None if pd.isnull(row['Keyword']) else row['Keyword'])
    def __getitem__(self, item):
        return {"Id":self.Id[item],"Title":self.Title[item],"Year":self.Year[item],"ConferenceId":self.ConferenceId[item],"JournalId":self.JournalId[item],"Keyword":self.Keyword[item]}
    def __len__(self):
        return len(self.Id)


