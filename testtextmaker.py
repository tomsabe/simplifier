"""
Load a JSON "gametext" file with items from ASSET test set

"""

# ISSUES

import dataloader

tdf = dataloader.load_asset_test()
orig = tdf['test.orig']
orig.to_json('testset.json')
