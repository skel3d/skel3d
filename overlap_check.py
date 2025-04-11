import os
import json
import math

orig_root = '/nas/exchange/objaverse/objaverse/'
with open(os.path.join(orig_root, 'valid_paths.json')) as f:
    orig_paths = json.load(f)
    total_objects = len(orig_paths)
    print('total_objects orig', total_objects)
    orig_paths_val = orig_paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
    orig_paths_train = orig_paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training


new_root = '../objaverse-rendering/rendering/objaverse_skel/'
with open(os.path.join(new_root, 'valid_paths.json')) as f:
    new_paths = json.load(f)
    total_objects = len(new_paths)    
    print('total_objects new', total_objects)    
    new_paths_val = new_paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
    new_paths_train = new_paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training





val_set = set(orig_paths_val) & set(new_paths)

train_set = set(new_paths) - val_set





# Save candidate val list as val.json
with open('val.json', 'w') as f:
    json.dump(list(val_set), f)

# Save orig_paths_train and new_paths_train as train.json
with open('train.json', 'w') as f:
    json.dump(list(train_set), f)


with open('train.json') as f:
    new_pathsaa = json.load(f)
    print('new_paths', len(new_pathsaa))


with open('val.json') as f:
    new_pathsaa = json.load(f)
    print('valalala', len(new_pathsaa))




    