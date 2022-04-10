from cmath import log
from distutils.log import Log
from module.trainer import Trainer
from module.logger import Logger

if __name__ == "__main__":
    import yaml
    from module.config import Config
    args = Config(yaml.load(open("config.yaml"), Loader=yaml.SafeLoader))
    logger = Logger(args)
    try:
        trainer = Trainer(args)
        trainer.train()
    except Exception as e:
        logger.error(str(e))
        raise Exception(str(e))