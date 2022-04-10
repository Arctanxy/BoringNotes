import yaml

class Config(dict):
    def __getattr__(self, __name: str) -> None:
        try:
            return self[__name]
        except:
            raise AttributeError(__name)

    def __setattr__(self, __name, __value) -> None:
        self[__name] = __value
        
def load_args(yml_file):
    f = open(yml_file)
    params = yaml.load(f, Loader=yaml.SafeLoader)
    args = Config(params)
    return args