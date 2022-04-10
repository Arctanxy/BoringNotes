import imp
import os 
import sys
import yaml
from glob import glob
from module.config import Config

yml_file = sys.argv[1]
f = open(yml_file)
params = yaml.load(f, Loader=yaml.SafeLoader)
args = Config(params)
print("args ", args)
def gen_vocablist(args):
    text_files = glob(os.path.join(args.text_folder, "*txt_bak"))
    all_words = ""
    for file in text_files:
        try:
            content = open(file, 'r', encoding="utf-8").read()
        except:
            content = open(file, 'r', encoding="gbk").read()
        all_words += content
    unique_list = list(set(all_words)) #  + "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ~`!@#$%^&*()_+-={}|[]\\:\"<>?;',./'"))
    with open(args.vocab_list, 'w', encoding="utf-8") as f:
        string = "".join(unique_list).replace("\n", "").replace(" ", "").replace("\t", "")
        f.write(string)
        print("size of vocab list is ", len(string))
    return unique_list

if __name__ == "__main__":
    gen_vocablist(args)
