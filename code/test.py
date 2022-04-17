from collections import defaultdict
dic = defaultdict(dict)
dic[1] = {'author':[]}
print(dic[1].get('ConferenceId'))
from torch.utils.data import Dataset
class test_dataset(Dataset):
    def __init__(self):
        self.data = [[1],[2],[3],[46765756],[756767],[3657567]]
        self.label = [1,1,1,0,0,0]
    def __getitem__(self, item):
        return self.data[item],self.y[item]
    def __len__(self):
        return len(self.data)


