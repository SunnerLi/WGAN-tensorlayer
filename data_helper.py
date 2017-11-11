from download import download_mnist, download_celeb_a, download_lsun
from glob import glob
import numpy as np
import cv2
import os

class ImageHandler(object):
    def __init__(self, dataset_name='mnist', batch_size=32):
        """
            dataset_name can be:
                1. mnist
                2. lsun
                3. celeba
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.batch_counter = 0
        self.image_tensor = None
        self.download()        

    def download(self):
        if not os.path.exists('data/'):
            os.mkdir('data')
        if self.dataset_name == 'lsun':
            if not os.path.exists('data/lsun'):
                download_lsun('data')
                os.system('unzip data/lsun/church_outdoor_train_lmdb.zip')
                os.system('python3 wrapper/lsun/data.py export church_outdoor_train_lmdb --out_dir data/lsun')
            self.image_tensor = glob('data/lsun/*.webp')
        elif self.dataset_name == 'mnist':
            if not os.path.exists('data/mnist'):
                download_mnist('data')
            self.image_tensor = self.load_mnist()
        elif self.dataset_name == 'celeba':
            if not os.path.exists('data/celeba'):
                download_celeb_a('data')
            self.image_tensor = glob('data/celebA/*.jpg')
        else:
            raise Exception('invalid dataset name...')

    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)
        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        X = loaded[16:].reshape((60000,28,28,1)).astype(np.float)
        return X

    def getBatchImage(self):
        result = self.image_tensor[self.batch_counter * self.batch_size : self.batch_counter * self.batch_size + self.batch_size]
        if self.dataset_name != 'mnist':
            result = [cv2.resize(cv2.imread(img_name), (28, 28)) for img_name in result]
            result = np.asarray(result, dtype=np.float)
        if (self.batch_counter + 1) == len(self.image_tensor) // self.batch_size:
            self.batch_counter = 0
        else:
            self.batch_counter += 1
        return result/255.

def generateNoice(batch_size, dim):
    return np.random.normal(loc=0.0, scale=1.0, size=[batch_size, dim])

if __name__ == '__main__':
    handler = ImageHandler('lsun')
    print(handler.getBatchImage())