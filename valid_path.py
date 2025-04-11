import os
import json


folder_path = '/objaverse/'
valid_paths = []

valid_paths = os.listdir(folder_path)

print('============= length of dataset %d =============' % len(valid_paths))

# read train and val json files and compare the content with valid_paths
# if the content is in valid_paths, then append the content to train_v.jsom and val_v.json

train_v = []
# load the file name
with open(os.path.join('train.json')) as f:
    train_paths = json.load(f)
    print('============= length of orig dataset %d =============' % len(train_paths))
    # compare the content with valid_paths. as set is faster than list
    train_paths_set = set(train_paths)
    valid_paths_set = set(valid_paths)
    # find the intersection
    intersection = train_paths_set.intersection(valid_paths_set)
    train_v = list(intersection)
    print('============= length of new train dataset %d =============' % len(train_v))
    # save it to train_v.json
    output_file = 'train_v.json'
    with open(output_file, 'w') as f:
        json.dump(train_v, f)

val_v = []
# load the file name
with open(os.path.join('val.json')) as f:
    val_paths = json.load(f)
    print('============= length of dataset %d =============' % len(val_paths))
    # compare the content with valid_paths. as set is faster than list
    val_paths_set = set(val_paths)
    valid_paths_set = set(valid_paths)
    # find the intersection
    intersection = val_paths_set.intersection(valid_paths_set)
    val_v = list(intersection)
    print('============= length of dataset %d =============' % len(val_v))
    # save it to val_v.json
    output_file = 'val_v.json'
    with open(output_file, 'w') as f:
        json.dump(val_v, f)

