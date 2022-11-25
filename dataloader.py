""" Load validation or test data"""

import pandas as pd

PATH = './data/asset-main/dataset/'

def load_asset_valid_orig():
    """Return a DataFrame of the original text in validation set."""
    file = PATH+'asset.valid.orig'
    df = pd.read_table(file,names=['valid.orig'])
    return df

def load_asset_valid():
    """Return a DataFrame of original plus all simplifications, incl. GPT-3 as eleventh."""
    df = load_asset_valid_orig()
    for i in range(11):
        name = 'valid.simp.'+str(i)
        file = PATH+'asset.'+name
        col = pd.read_table(file,names=[name])
        df = pd.concat([df,col],axis=1)
    return df

def load_asset_test():
    file = PATH+'asset.test.orig'
    df = pd.read_table(file,names=['test.orig'])
    for i in range(10):
        name = 'test.simp.'+str(i)
        file = PATH+'asset.'+name
        col = pd.read_table(file,names=[name])
        df = pd.concat([df,col],axis=1)
    return df

if __name__ == '__main__':
    print("Loading asset-main validation set:\n")
    vdf = load_asset_valid()
    print(vdf)
    print("Loading asset-main test set:\n")
    tdf = load_asset_test()
    print(tdf)

