from pandas import DataFrame,read_csv,concat
from torch import from_numpy,load
import gdown
from os.path import expanduser
import os
from tqdm import tqdm
import tarfile

class nursing:
    def load_concatenated_data():
        url = "https://drive.google.com/uc?id=1sUONGddK-5eMhLnVizZzb2vPkb4t99Zd"
        filename = "nursing_concatenated.tar.gz"
        home = expanduser("~")
        outdir = f'{home}/.delta'
        filepath = f'{outdir}/{filename}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if os.path.exists(filepath):
            print(f'Already downloaded')
        else:
            gdown.download(url, filepath, quiet=False)

        featurefilename = f'X.pt'
        labelfilename = f'y.pt'
        featurefilepath = f'{outdir}/{featurefilename}'
        labelfilepath = f'{outdir}/{labelfilename}'
        
        with tarfile.open(filepath) as tar:
            tar.extractall(outdir)
        return load(f=featurefilepath),load(f=labelfilepath)
    def load_nursing_list(nurses):
        url = "https://drive.google.com/uc?id=1ZPVqr3cYLfR7i3fzwA0WSGbmeGVieP0D"
        filename = "nursing.tar.gz"
        home = expanduser("~")
        outdir = f'{home}/.delta'
        filepath = f'{outdir}/{filename}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if os.path.exists(filepath):
            print(f'Already downloaded')
        else:
            gdown.download(url, filepath, quiet=False)
        with tarfile.open(filepath) as tar:
            tar.extractall(outdir)
        X = DataFrame()
        y = DataFrame()
        for i in tqdm(nurses):
            Xi = read_csv(f'{outdir}/nursing/{i}.csv')
            yi = Xi.pop('label')
            # should really window here to avoid overlap
            X = concat([X,Xi])
            y = concat([y,yi])
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        X = from_numpy(X.values).float()
        y = from_numpy(y.values).squeeze().long()
        return X,y
