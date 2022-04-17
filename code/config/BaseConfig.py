import yaml
class BaseConfig:
    def __init__(self,config_path):
        fs = open(config_path, 'r',encoding="UTF-8")
        self.config = yaml.load(fs, Loader=yaml.FullLoader)
        print(self.config['FilePath']['author'])