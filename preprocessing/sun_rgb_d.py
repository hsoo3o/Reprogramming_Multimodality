import scipy.io
import os

import numpy as np
import torch
from PIL import Image

# gt: type of the classes
# target: gt label of each data

# get gt classes
def get_gt(root):
    train_scene_name = {}
    split_info = scipy.io.loadmat(os.path.join(root,'allsplit.mat'))

    for train_path in split_info['alltrain'][0]:
        train_path = root +'/'+ '/'.join(train_path.item().split('/')[6:])
        with open (train_path + '/scene.txt') as f:
            tmp_s = f.read()
            try:
                train_scene_name[tmp_s] += 1
            except:
                train_scene_name[tmp_s] = 1

    tmp = dict(sorted(train_scene_name.items(), key=lambda item: item[1], reverse=True)[:19]).keys()
    gt_classes = sorted(list(tmp))

    return gt_classes



def get_data(root,gt_classes,tag):
    '''
    if train -> tag: alltrain
    if test -> tag: alltest
    SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat -> SUNRGBD/allsplit.mat
    '''
    split_info = scipy.io.loadmat(os.path.join(root,'allsplit.mat'))

    depth_path = []
    image_path = []
    target_list = []

    for path in split_info[tag][0]:
        path = root +'/'+ '/'.join(path.item().split('/')[6:])
        with open (path + '/scene.txt') as f:
            tmp_s = f.read()
            if tmp_s in gt_classes:
                target_list.append(tmp_s)
                depth_path.append(path + '/depth_bfx')
                image_path.append(path + '/image')
    
    return depth_path, image_path, target_list

#  depth to disparity



def convert_depth_to_disparity(depth_path, min_depth=0.01, max_depth=75.0):
    """
    depth_file is a png file that contains the scene depth
    intrinsics_file is a txt file supplied in SUNRGBD with sensor information
            Can be found at the path: os.path.join(root_dir, room_name, "intrinsics.txt")
    # code from omnivore
    """
    sensor_to_params = {
        "kv1": {
            "baseline": 0.075,
        },
        "kv1_b": {
            "baseline": 0.075,
        },
        "kv2": {
            "baseline": 0.075,
        },
        "realsense": {
            "baseline": 0.095,
        },
        "xtion": {
            "baseline": 0.095, # guessed based on length of 18cm for ASUS xtion v1
        },
    }
    
    depth_file = os.path.join(depth_path, os.listdir(depth_path)[0])
    intrinsics_file = os.path.join('/'.join(depth_path.split('/')[:-1]), 'intrinsics.txt')
    sensor_type = depth_path.split('/')[8]
    with open(intrinsics_file, 'r') as fh:
        lines = fh.readlines()
        focal_length = float(lines[0].strip().split()[0])
    baseline = sensor_to_params[sensor_type]["baseline"]
    depth_image = np.array(Image.open(depth_file))
    depth = np.array(depth_image).astype(np.float32)
    depth_in_meters = depth / 1000. 
    if min_depth is not None:
        depth_in_meters = depth_in_meters.clip(min=min_depth, max=max_depth)
    disparity = baseline * focal_length / depth_in_meters
    disparity = disparity / max_depth
    return torch.from_numpy(disparity).float()


# data preprocess

def get_preprocessed_img_data(image_path, preprocess):
    
    processed_img = []

    for folder_path in image_path:
        file_name = os.listdir(folder_path)
        path = os.path.join(folder_path, file_name[0])
        with open(path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = preprocess(image)   
        processed_img.append(image)
        
    image_input = torch.stack(processed_img)

    return image_input

def get_preprocessed_depth_data(depth_path, preprocess):
    
    processed_disparity = []
    
    for path in depth_path:
        disparity = convert_depth_to_disparity(path,max_depth=255)
        disparity = disparity.unsqueeze(0) 

        processed_disparity.append(preprocess(disparity))
    disparity_input = torch.stack(processed_disparity)

    return disparity_input



# get target
def get_target(target_list, gt_classes):

    target = []
    for t in target_list:
        target.append(list(gt_classes).index(t))
    target = torch.tensor(target)

    return target