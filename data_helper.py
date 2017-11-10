import numpy as np
import os

class ImageHandler(object):
    def __init__(self, dataset_name='mnist'):
        """
            dataset_name can be:
                1. mnist
                2. lsun
        """
        if dataset_name == 'lsun':
            if not os.path.exists('data/'):
                os.mkdir('data')
            # os.system('python lsun/download.py -c church_outdoor')
            # os.system('unzip church_outdoor_train_lmdb.zip')
            os.system('python3 lsun/data.py export church_outdoor_train_lmdb --out_dir data')
        else:
            raise Exception('invalid dataset name...')

def generateNoice(batch_size, dim):
    return np.random.normal(loc=0.0, scale=1.0, size=[batch_size, dim])

handler = ImageHandler('lsun')