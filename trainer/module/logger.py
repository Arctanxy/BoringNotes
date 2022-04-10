import os 
import time

class Logger:
    def __init__(self, args) -> None:
        self.args = args
        self.logfile = os.path.join(self.args.log_dir, "log.txt")
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
    
    def info(self, string):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
        with open(self.logfile, 'a') as f:
            f.write("[Info]{} - {}\n".format(current_time, str(string)))
    
    def warn(self, string):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
        with open(self.logfile, 'a') as f:
            f.write("[Warning]{} - {}\n".format(current_time, str(string)))
        
    def error(self, string):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
        with open(self.logfile, 'a') as f:
            f.write("[Error]{} - {}\n".format(current_time, str(string)))
        

        

