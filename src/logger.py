





import logging
import sys

class easyLog:
    def __init__(self,file_name):
        logging.basicConfig(filename=file_name,
                    format='%(asctime)s %(message)s',
                    filemode='w',level=logging.INFO,encoding='utf-8')
        
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        self.stdout = logging.StreamHandler(stream=sys.stdout)
        self.stdout.setLevel(logging.INFO)

        self.logger.addHandler(self.stdout)



    def info(self,text:str):
        self.logger.info(text + "\n")