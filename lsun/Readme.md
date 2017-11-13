LSUN Dataset Wrapper
---

This folder collects the script with new python version(3+). The original version is [here](https://github.com/fyu/lsun). I merge `download.py` which is written by [dalematt](https://github.com/fyu/lsun/issues/5). The generated images will locate in `../data/lsun/` folder.    

Usage
---
#### Download data
```bash
# Download the whole latest data set
python2.7 download.py
# Download the whole latest data set to <data_dir>
python2.7 download.py -o <data_dir>
# Download data for bedroom
python2.7 download.py -c bedroom
# Download testing set
python2.7 download.py -c test
```

#### View the lmdb content
```bash
python2.7 data.py view <image db path>
```

#### Export the images to a folder
```bash
python2.7 data.py export <image db path> --out_dir <output directory>
```

#### Example:
Export all the images in valuation sets in the current folder to a
"data"
subfolder.

```bash
python2.7 data.py export *_val_lmdb --out_dir data
```