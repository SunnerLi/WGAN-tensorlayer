import argparse
import numpy
import lmdb
import cv2
import os

__author__ = 'Fisher Yu'
__email__ = 'fy@cs.princeton.edu'
__license__ = 'MIT'
__modified__ = 'SunnerLi'

"""
    This script revises the byte name problem, and only permit to transfer the whole images into single folder
"""

def view(db_path):
    print('Viewing', db_path)
    print('Press ESC to exist or SPACE to advance.')
    window_name = 'LSUN'
    cv2.namedWindow(window_name)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            print('Current key:', key)
            img = cv2.imdecode(
                numpy.fromstring(val, dtype=numpy.uint8), 1)
            cv2.imshow(window_name, img)
            c = cv2.waitKey()
            if c == 27:
                break

def export_images(db_path, out_dir, limit=10):
    print('Exporting', db_path, 'to', out_dir)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            if out_dir[-1] == '/':
                image_out_path = out_dir + key[:6].decode('utf-8') + '.webp'
            else:
                image_out_path = out_dir + '/' + key[:6].decode('utf-8') + '.webp'
            with open(image_out_path, 'wb') as fp:
                fp.write(val)
            count += 1
            if count == limit:
                break
            if count % 1000 == 0:
                print('Finished', count, 'images')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', nargs='?', type=str,
                        choices=['view', 'export'],
                        help='view: view the images in the lmdb database '
                             'interactively.\n'
                             'export: Export the images in the lmdb databases '
                             'to a folder. The images are grouped in subfolders'
                             ' determinted by the prefiex of image key.')
    parser.add_argument('lmdb_path', nargs='+', type=str,
                        help='The path to the lmdb database folder. '
                             'Support multiple database paths.')
    parser.add_argument('--out_dir', type=str, default='')
    args = parser.parse_args()
    for lmdb_path in args.lmdb_path:
        if args.command == 'view':
            view(lmdb_path)
        elif args.command == 'export':
            export_images(lmdb_path, args.out_dir)