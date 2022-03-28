import logging  # 引入logging模块
import os.path
import time
# 第一步，创建一个logger
import sys
def get_logger(log_level=logging.INFO,log_file="log.log"):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        # rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        # log_path = os.path.dirname(os.getcwd()) + '/Logs/'
        # log_name = log_path + rq + '.log'
        # logfile = log_name
        formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt="%Y/%m/%d %H:%M:%S")
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if not os.path.exists(log_file):
            f = open(log_file,"w",encoding='utf-8')
            f.close()
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        #fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        # formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        # fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        logger.addHandler(file_handler)
        logger.setLevel(log_level)
        # 日志
        # return logger
        # logger.debug('this is a logger debug message')
        # logger.info('this is a logger info message')
        # logger.warning('this is a logger warning message')
        # logger.error('this is a logger error message')
        # logger.critical('this is a logger critical message')
        return logger

# logger = get_logger()

# logger.debug('this is a logger debug message')
#logger.info('this is a logger info message')
# logger.warning('this is a logger warning message')
# logger.error('this is a logger error message')
# logger.critical('this is a logger critical message')

# logging.basicConfig(level=logging.NOTSET)  # 设置日志级别
# logging.debug(u"如果设置了日志级别为NOTSET,那么这里可以采取debug、info的级别的内容也可以显示在控制台上了")
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)  # Log等级总开关
# 第二步，创建一个handler，用于写入日志文件
# rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
# log_path = os.path.dirname(os.getcwd()) + '/Logs/'
# log_name = log_path + rq + '.log'
# logfile = log_name
