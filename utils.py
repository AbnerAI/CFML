import shutil
import logging
import os
import datetime
def backup_codes(path):
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.realpath(os.path.join(current_path, os.pardir))

    path = os.path.join(path, "codes_{}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    os.makedirs(path, exist_ok=True)

    names = ["models"]
    for name in names:
        if os.path.exists(os.path.join(root_path, name)):
            shutil.copytree(os.path.join(root_path, name), os.path.join(path, name))

    pyfiles = filter(lambda x: x.endswith(".py"), os.listdir(root_path))
    for pyfile in pyfiles:
        shutil.copy(os.path.join(root_path, pyfile), os.path.join(path, pyfile))

def set_logger(path,file_path=None):
    os.makedirs(path,exist_ok=True)
    #logger to print information
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler1 = logging.StreamHandler()
    if file_path is not None:
        handler2 = logging.FileHandler(os.path.join(path,file_path), mode='w')
    else:
        handler2 = logging.FileHandler(os.path.join(path, "logs.txt"), mode='w')
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)