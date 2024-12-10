import scipy.io
import h5py
import numpy as np
from PIL import Image
import torch

# data load

def get_depth_n_image(file_path):
    with h5py.File(file_path, 'r') as f:
        depths = f['depths'][:]
        images = f['images'][:]
    
    return depths, images

def get_train_n_test_idx(file_path):

    train_test_split = scipy.io.loadmat(file_path)

    train_idx = [idx.item() - 1 for idx in train_test_split['trainNdxs']]
    test_idx = [idx.item() - 1 for idx in train_test_split['testNdxs']]

    return train_idx, test_idx

def read_depth_name(file_path,dataset_name):

    with h5py.File(file_path, 'r') as f:
        if dataset_name in f:
            names_refs = f[dataset_name][:]
            names_list = []
            
            for ref in names_refs.flatten():
                if isinstance(ref, h5py.Reference):
                    dataset = f[ref]
                    
                    if isinstance(dataset, h5py.Dataset):
                        ascii_values = dataset[:]
                        decoded_string = ''.join(chr(val[0]) for val in ascii_values)
                        names_list.append(decoded_string)
    
    return np.asarray(names_list)


def get_data(data_file_path, data_idx):

    all_d,all_i = get_depth_n_image(data_file_path)
    # train_idx, test_idx = get_train_n_test_idx(split_file_path)

    gt_type_list = read_depth_name(data_file_path, 'sceneTypes')

    data = [all_d[data_idx], all_i[data_idx], gt_type_list[data_idx]]

    return data


# data preprocess

def get_preprocessed_img_data(raw_image_data, preprocess):
    processed_img = []
    for i in range(raw_image_data.shape[0]):
        rgb_image = np.transpose(raw_image_data[i, :, :, :], (2, 1, 0))  
        img = Image.fromarray(rgb_image)
        
        processed_img.append(preprocess(img))
        
    image_input = torch.tensor(np.stack(processed_img))

    return image_input



def depth_to_disparity(depth_map, focal_length=518.857901, baseline=0.07):

    depth_map[depth_map == 0] = 1e-6
    disparity_map = (focal_length * baseline) / depth_map
    return disparity_map

def get_preprocessed_depth_data(raw_depth_data, preprocess):

    disparities = depth_to_disparity(raw_depth_data)
    processed_disparity = []
    for i in range(disparities.shape[0]):

        disparity_map = np.transpose(disparities[i, :, :], (1, 0))
        disparity = torch.tensor(disparity_map)
        disparity = disparity.clamp(min=0.01)

        disparity /= (225*2)
        disparity = disparity.unsqueeze(0) 

        processed_disparity.append(preprocess(disparity))
    disparity_input = torch.tensor(np.stack(processed_disparity))

    return disparity_input


# get gt classes
def get_gt(data_gt_list):
    return set(data_gt_list)


# get target
def get_target(file_path, data_idx, gt_classes):
    target_list = [name.split('_0')[0] for name in read_depth_name(file_path, 'scenes')]

    target = []
    for t in target_list:
        target.append(list(gt_classes).index(t))
    target = torch.tensor(target)

    return target[data_idx]