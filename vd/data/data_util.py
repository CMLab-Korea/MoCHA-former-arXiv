import os
import os.path as osp
import glob
import cv2
import numpy as np

from basicsr.utils import scandir



def vdemoire_data_loader_train(folders):
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    input_folder, gt_folder = folders

    gt_paths = list(scandir(gt_folder))
    gt_names = []
    for gt_path in gt_paths:
        gt_name = osp.basename(gt_path).split('.jpg')[0]
        gt_names.append(gt_name)

    paths = []
    for gt_name in gt_names:
        gt_path = osp.join(gt_folder, gt_name + '.jpg')
        scene_idx = gt_name.split('_')[0]
        patch_idx = gt_name[-4:]

        lq_1_idx = int(gt_name.split('_')[1])  # 42
        if lq_1_idx != 0 and lq_1_idx != 59:
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)
        elif lq_1_idx == 0:
            lq_0_name = gt_name  # 0
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)  # 1
        else:
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)  # 58
            lq_2_name = gt_name  # 59

        lq_0_path = osp.join(input_folder, lq_0_name + '.jpg')
        lq_1_path = osp.join(input_folder, gt_name + '.jpg')
        lq_2_path = osp.join(input_folder, lq_2_name + '.jpg')
        paths.append(dict(
            [('lq_0_path', lq_0_path), ('lq_1_path', lq_1_path), ('lq_2_path', lq_2_path), ('gt_path', gt_path),
             ('key', gt_name)]))
    return paths


def vdemoire_data_loader_test(folders):
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    input_folder, gt_folder = folders

    gt_paths = list(scandir(gt_folder))
    gt_names = []
    for gt_path in gt_paths:
        gt_name = osp.basename(gt_path).split('.jpg')[0]
        gt_names.append(gt_name)

    paths = []
    for gt_name in gt_names:
        gt_path = osp.join(gt_folder, gt_name + '.jpg')
        scene_idx = gt_name.split('_')[0]

        lq_1_idx = int(gt_name.split('_')[1])  # 42
        if lq_1_idx != 0 and lq_1_idx != 59:
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)
        elif lq_1_idx == 0:
            lq_0_name = gt_name  # 0
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)  # 1
        else:
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5) # 58
            lq_2_name = gt_name  # 59

        lq_0_path = osp.join(input_folder, lq_0_name + '.jpg')
        lq_1_path = osp.join(input_folder, gt_name + '.jpg')
        lq_2_path = osp.join(input_folder, lq_2_name + '.jpg')
        paths.append(dict(
            [('lq_0_path', lq_0_path), ('lq_1_path', lq_1_path), ('lq_2_path', lq_2_path), ('gt_path', gt_path),
             ('key', gt_name)]))
    return paths

def video_rawrgb_3frames_from_folders_train(folders):
    assert len(folders) == 2, (
    'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    # input_folder, gt_folder = folders
    lq_raw_folder, gt_rgb_folder = folders

    gt_rgb_paths = list(scandir(gt_rgb_folder))
    gt_rgb_names = []
    for gt_rgb_path in gt_rgb_paths:
        gt_rgb_name = osp.basename(gt_rgb_path).split('.png')[0]
        gt_rgb_names.append(gt_rgb_name)

    paths = []
    for gt_rgb_name in gt_rgb_names:
        gt_rgb_path = osp.join(gt_rgb_folder, gt_rgb_name + '.png')
        scene_idx = gt_rgb_name.split('_')[0]
        patch_idx = gt_rgb_name[-2:]
        
        lq_1_idx = int(gt_rgb_name.split('_')[1])
        if lq_1_idx != 1 and lq_1_idx != 60: # when center frame is not the first and last frame
            # frame number: (lq_0_name, lq_1_name, lq_2_name) = (t-1, t, t+1)
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
        elif lq_1_idx == 1: # when center frame is the first frame
            lq_0_name = gt_rgb_name
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
        elif lq_1_idx == 60: # when center frame is the last frame
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            lq_2_name = gt_rgb_name
        else: 
            print("target frame has incorrect frame_number:")
            print("frmae_number: ", lq_1_idx)

        # save moired raw image paths
        lq_raw_0_path = osp.join(lq_raw_folder, lq_0_name + '.npz')
        lq_raw_1_path = osp.join(lq_raw_folder, gt_rgb_name + '.npz')
        lq_raw_2_path = osp.join(lq_raw_folder, lq_2_name + '.npz')
        
        paths.append(dict([
            ('lq_raw_0_path', lq_raw_0_path), 
            ('lq_raw_1_path', lq_raw_1_path), 
            ('lq_raw_2_path', lq_raw_2_path), 
            ('gt_rgb_path', gt_rgb_path),
            ('key', gt_rgb_name)]))

    return paths

def video_rawrgb_3frames_from_folders_val(folders):
    assert len(folders) == 2, (
    'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    # input_folder, gt_folder = folders
    lq_raw_folder,gt_rgb_folder = folders

    gt_rgb_paths = list(scandir(gt_rgb_folder))
    gt_rgb_names = []
    for gt_rgb_path in gt_rgb_paths:
        gt_rgb_name = osp.basename(gt_rgb_path).split('.png')[0]
        gt_rgb_names.append(gt_rgb_name)

    paths = []
    for gt_rgb_name in gt_rgb_names:
        gt_rgb_path = osp.join(gt_rgb_folder, gt_rgb_name + '.png')
        scene_idx = gt_rgb_name.split('_')[0]
        
        lq_1_idx = int(gt_rgb_name.split('_')[1])
        if lq_1_idx != 1 and lq_1_idx != 60: # when center frame is not the first and last frame
            # frame number: (lq_0_name, lq_1_name, lq_2_name) = (t-1, t, t+1)
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
        elif lq_1_idx == 1: # when center frame is the first frame
            lq_0_name = gt_rgb_name
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
        elif lq_1_idx == 60: # when center frame is the last frame
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            lq_2_name = gt_rgb_name
        else: 
            print("target frame has incorrect frame_number:")
            print("frmae_number: ", lq_1_idx)

        # save moired raw image paths
        lq_raw_0_path = osp.join(lq_raw_folder, lq_0_name + '.npz')
        lq_raw_1_path = osp.join(lq_raw_folder, gt_rgb_name + '.npz')
        lq_raw_2_path = osp.join(lq_raw_folder, lq_2_name + '.npz')
        
        paths.append(dict([
            ('lq_raw_0_path', lq_raw_0_path), 
            ('lq_raw_1_path', lq_raw_1_path), 
            ('lq_raw_2_path', lq_raw_2_path), 
            ('gt_rgb_path', gt_rgb_path),
            ('key', gt_rgb_name)]))

    return paths

def multiframe_paired_paths_from_folders_val(folders):
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    input_folder, gt_folder = folders

    gt_paths = list(scandir(gt_folder))
    gt_names = []
    for gt_path in gt_paths:
        gt_name = osp.basename(gt_path).split('.jpg')[0]
        gt_names.append(gt_name)

    paths = []
    for gt_name in gt_names:
        gt_path = osp.join(gt_folder, gt_name + '.jpg')
        scene_idx = gt_name.split('_')[0]

        lq_1_idx = int(gt_name.split('_')[1])  # 42
        if lq_1_idx != 0 and lq_1_idx != 59:
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)
        elif lq_1_idx == 0:
            lq_0_name = gt_name  # 0
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)  # 1
        else:
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5) # 58
            lq_2_name = gt_name  # 59

        lq_0_path = osp.join(input_folder, lq_0_name + '.jpg')
        lq_1_path = osp.join(input_folder, gt_name + '.jpg')
        lq_2_path = osp.join(input_folder, lq_2_name + '.jpg')
        paths.append(dict(
            [('lq_0_path', lq_0_path), ('lq_1_path', lq_1_path), ('lq_2_path', lq_2_path), ('gt_path', gt_path),
             ('key', gt_name)]))
    return paths

def video_rawrgb_3frames_from_folders_train_inference(folders):
    assert len(folders) == 2, (
    'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    # input_folder, gt_folder = folders
    lq_raw_folder, gt_rgb_folder = folders

    gt_rgb_paths = list(scandir(gt_rgb_folder))
    gt_rgb_names = []
    for gt_rgb_path in gt_rgb_paths:
        gt_rgb_name = osp.basename(gt_rgb_path).split('.png')[0]
        gt_rgb_names.append(gt_rgb_name)

    paths = []
    for gt_rgb_name in gt_rgb_names:
        gt_rgb_path = osp.join(gt_rgb_folder, gt_rgb_name + '.png')
        scene_idx = gt_rgb_name.split('_')[0]
        patch_idx = gt_rgb_name[-2:]
        
        lq_1_idx = int(gt_rgb_name.split('_')[1])
        if lq_1_idx != 1 and lq_1_idx != 60: # when center frame is not the first and last frame
            # frame number: (lq_0_name, lq_1_name, lq_2_name) = (t-1, t, t+1)
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
        elif lq_1_idx == 1: # when center frame is the first frame
            lq_0_name = gt_rgb_name
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
        elif lq_1_idx == 60: # when center frame is the last frame
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            lq_2_name = gt_rgb_name
        else: 
            print("target frame has incorrect frame_number:")
            print("frmae_number: ", lq_1_idx)

        # save moired raw image paths
        lq_raw_0_path = osp.join(lq_raw_folder, lq_0_name + '.npz')
        lq_raw_1_path = osp.join(lq_raw_folder, gt_rgb_name + '.npz')
        lq_raw_2_path = osp.join(lq_raw_folder, lq_2_name + '.npz')
        
        paths.append(dict([
            ('lq_raw_0_path', lq_raw_0_path), 
            ('lq_raw_1_path', lq_raw_1_path), 
            ('lq_raw_2_path', lq_raw_2_path), 
            ('gt_rgb_path', gt_rgb_path),
            ('key', gt_rgb_name)]))

    return paths

def video_rawrgb_3frames_from_folders_val_inference(folders):
    assert len(folders) == 2, (
    'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    # input_folder, gt_folder = folders
    lq_raw_folder,gt_rgb_folder = folders

    gt_rgb_paths = list(scandir(gt_rgb_folder))
    gt_rgb_names = []
    for gt_rgb_path in gt_rgb_paths:
        gt_rgb_name = osp.basename(gt_rgb_path).split('.png')[0]
        gt_rgb_names.append(gt_rgb_name)

    paths = []
    for gt_rgb_name in gt_rgb_names:
        gt_rgb_path = osp.join(gt_rgb_folder, gt_rgb_name + '.png')
        scene_idx = gt_rgb_name.split('_')[0]
        
        lq_1_idx = int(gt_rgb_name.split('_')[1])
        if lq_1_idx != 1 and lq_1_idx != 60: # when center frame is not the first and last frame
            # frame number: (lq_0_name, lq_1_name, lq_2_name) = (t-1, t, t+1)
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
        elif lq_1_idx == 1: # when center frame is the first frame
            lq_0_name = gt_rgb_name
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
        elif lq_1_idx == 60: # when center frame is the last frame
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            lq_2_name = gt_rgb_name
        else: 
            print("target frame has incorrect frame_number:")
            print("frmae_number: ", lq_1_idx)

        # save moired raw image paths
        lq_raw_0_path = osp.join(lq_raw_folder, lq_0_name + '.npz')
        lq_raw_1_path = osp.join(lq_raw_folder, gt_rgb_name + '.npz')
        lq_raw_2_path = osp.join(lq_raw_folder, lq_2_name + '.npz')
        
        paths.append(dict([
            ('lq_raw_0_path', lq_raw_0_path), 
            ('lq_raw_1_path', lq_raw_1_path), 
            ('lq_raw_2_path', lq_raw_2_path), 
            ('gt_rgb_path', gt_rgb_path),
            ('key', gt_rgb_name)]))

    return paths


def video_rawrgb_3_to_3_frames_from_folders_train(folders):
    assert len(folders) == 2, (
    'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    # input_folder, gt_folder = folders
    lq_raw_folder,gt_rgb_folder = folders

    gt_rgb_paths = list(scandir(gt_rgb_folder))
    gt_rgb_names = []
    for gt_rgb_path in gt_rgb_paths:
        gt_rgb_name = osp.basename(gt_rgb_path).split('.png')[0]
        gt_rgb_names.append(gt_rgb_name)

    paths = []
    for gt_rgb_name in gt_rgb_names:
        gt_rgb_path = osp.join(gt_rgb_folder, gt_rgb_name + '.png')
        scene_idx = gt_rgb_name.split('_')[0]
        
        
        lq_1_idx = int(gt_rgb_name.split('_')[1])
        if lq_1_idx != 1 and lq_1_idx != 60: # when center frame is not the first and last frame
            # frame number: (lq_0_name, lq_1_name, lq_2_name) = (t-1, t, t+1)
            gt_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            gt_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
            
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
            
        elif lq_1_idx == 1: # when center frame is the first frame
            gt_0_name = gt_rgb_name
            gt_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
            
            lq_0_name = gt_rgb_name
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
            
        elif lq_1_idx == 60: # when center frame is the last frame
            gt_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            gt_2_name = gt_rgb_name
            
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            lq_2_name = gt_rgb_name
            
        else: 
            print("target frame has incorrect frame_number:")
            print("frmae_number: ", lq_1_idx)

        # save gt image paths
        gt_rgb_0_path = osp.join(gt_rgb_folder, gt_0_name + '.png')
        gt_rgb_1_path = osp.join(gt_rgb_folder, gt_rgb_name + '.png')
        gt_rgb_2_path = osp.join(gt_rgb_folder, gt_2_name + '.png')
        
        # save moired raw image paths
        lq_raw_0_path = osp.join(lq_raw_folder, lq_0_name + '.npz')
        lq_raw_1_path = osp.join(lq_raw_folder, gt_rgb_name + '.npz')
        lq_raw_2_path = osp.join(lq_raw_folder, lq_2_name + '.npz')
        
        paths.append(dict([
            ('lq_raw_0_path', lq_raw_0_path), 
            ('lq_raw_1_path', lq_raw_1_path), 
            ('lq_raw_2_path', lq_raw_2_path), 
            ('gt_rgb_0_path', gt_rgb_0_path),
            ('gt_rgb_1_path', gt_rgb_1_path),
            ('gt_rgb_2_path', gt_rgb_2_path),
            ('key', gt_rgb_name)]))

    return paths

def video_rawrgb_3_to_3_frames_from_folders_val(folders):
    assert len(folders) == 2, (
    'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    # input_folder, gt_folder = folders
    lq_raw_folder,gt_rgb_folder = folders

    gt_rgb_paths = list(scandir(gt_rgb_folder))
    gt_rgb_names = []
    for gt_rgb_path in gt_rgb_paths:
        gt_rgb_name = osp.basename(gt_rgb_path).split('.png')[0]
        gt_rgb_names.append(gt_rgb_name)

    paths = []
    for gt_rgb_name in gt_rgb_names:
        gt_rgb_path = osp.join(gt_rgb_folder, gt_rgb_name + '.png')
        scene_idx = gt_rgb_name.split('_')[0]
        
        
        lq_1_idx = int(gt_rgb_name.split('_')[1])
        if lq_1_idx != 1 and lq_1_idx != 60: # when center frame is not the first and last frame
            # frame number: (lq_0_name, lq_1_name, lq_2_name) = (t-1, t, t+1)
            gt_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            gt_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
            
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
            
        elif lq_1_idx == 1: # when center frame is the first frame
            gt_0_name = gt_rgb_name
            gt_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
            
            lq_0_name = gt_rgb_name
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
            
        elif lq_1_idx == 60: # when center frame is the last frame
            gt_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            gt_2_name = gt_rgb_name
            
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            lq_2_name = gt_rgb_name
            
        else: 
            print("target frame has incorrect frame_number:")
            print("frmae_number: ", lq_1_idx)

        # save gt image paths
        gt_rgb_0_path = osp.join(gt_rgb_folder, gt_0_name + '.png')
        gt_rgb_1_path = osp.join(gt_rgb_folder, gt_rgb_name + '.png')
        gt_rgb_2_path = osp.join(gt_rgb_folder, gt_2_name + '.png')
        
        # save moired raw image paths
        lq_raw_0_path = osp.join(lq_raw_folder, lq_0_name + '.npz')
        lq_raw_1_path = osp.join(lq_raw_folder, gt_rgb_name + '.npz')
        lq_raw_2_path = osp.join(lq_raw_folder, lq_2_name + '.npz')
        
        paths.append(dict([
            ('lq_raw_0_path', lq_raw_0_path), 
            ('lq_raw_1_path', lq_raw_1_path), 
            ('lq_raw_2_path', lq_raw_2_path), 
            ('gt_rgb_0_path', gt_rgb_0_path),
            ('gt_rgb_1_path', gt_rgb_1_path),
            ('gt_rgb_2_path', gt_rgb_2_path),
            ('key', gt_rgb_name)]))

    return paths


def n2n_for_test_train(folders):
    assert len(folders) == 2, (
    'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    # input_folder, gt_folder = folders
    lq_raw_folder,gt_rgb_folder = folders

    gt_rgb_paths = list(scandir(gt_rgb_folder))
    gt_rgb_names = []
    for gt_rgb_path in gt_rgb_paths:
        gt_rgb_name = osp.basename(gt_rgb_path).split('.png')[0]
        gt_rgb_names.append(gt_rgb_name)

    paths = []
    for gt_rgb_name in gt_rgb_names:
        gt_rgb_path = osp.join(gt_rgb_folder, gt_rgb_name + '.png')
        scene_idx = gt_rgb_name.split('_')[0]
        
        
        lq_1_idx = int(gt_rgb_name.split('_')[1])
        if lq_1_idx != 1 and lq_1_idx != 12: # when center frame is not the first and last frame
            # frame number: (lq_0_name, lq_1_name, lq_2_name) = (t-1, t, t+1)
            gt_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            gt_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
            
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
            
        elif lq_1_idx == 1: # when center frame is the first frame
            gt_0_name = gt_rgb_name
            gt_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
            
            lq_0_name = gt_rgb_name
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
            
        elif lq_1_idx == 12: # when center frame is the last frame
            gt_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            gt_2_name = gt_rgb_name
            
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            lq_2_name = gt_rgb_name
            
        else: 
            print("target frame has incorrect frame_number:")
            print("frmae_number: ", lq_1_idx)

        # save gt image paths
        gt_rgb_0_path = osp.join(gt_rgb_folder, gt_0_name + '.png')
        gt_rgb_1_path = osp.join(gt_rgb_folder, gt_rgb_name + '.png')
        gt_rgb_2_path = osp.join(gt_rgb_folder, gt_2_name + '.png')
        
        # save moired raw image paths
        lq_raw_0_path = osp.join(lq_raw_folder, lq_0_name + '.npz')
        lq_raw_1_path = osp.join(lq_raw_folder, gt_rgb_name + '.npz')
        lq_raw_2_path = osp.join(lq_raw_folder, lq_2_name + '.npz')
        
        paths.append(dict([
            ('lq_raw_0_path', lq_raw_0_path), 
            ('lq_raw_1_path', lq_raw_1_path), 
            ('lq_raw_2_path', lq_raw_2_path), 
            ('gt_rgb_0_path', gt_rgb_0_path),
            ('gt_rgb_1_path', gt_rgb_1_path),
            ('gt_rgb_2_path', gt_rgb_2_path),
            ('key', gt_rgb_name)]))

    return paths

def n2n_for_test_val(folders):
    assert len(folders) == 2, (
    'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    # input_folder, gt_folder = folders
    lq_raw_folder,gt_rgb_folder = folders

    gt_rgb_paths = list(scandir(gt_rgb_folder))
    gt_rgb_names = []
    for gt_rgb_path in gt_rgb_paths:
        gt_rgb_name = osp.basename(gt_rgb_path).split('.png')[0]
        gt_rgb_names.append(gt_rgb_name)

    paths = []
    for gt_rgb_name in gt_rgb_names:
        gt_rgb_path = osp.join(gt_rgb_folder, gt_rgb_name + '.png')
        scene_idx = gt_rgb_name.split('_')[0]
        
        
        lq_1_idx = int(gt_rgb_name.split('_')[1])
        if lq_1_idx != 1 and lq_1_idx != 12: # when center frame is not the first and last frame
            # frame number: (lq_0_name, lq_1_name, lq_2_name) = (t-1, t, t+1)
            gt_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            gt_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
            
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
            
        elif lq_1_idx == 1: # when center frame is the first frame
            gt_0_name = gt_rgb_name
            gt_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
            
            lq_0_name = gt_rgb_name
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(2)
            
        elif lq_1_idx == 12: # when center frame is the last frame
            gt_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            gt_2_name = gt_rgb_name
            
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(2)
            lq_2_name = gt_rgb_name
            
        else: 
            print("target frame has incorrect frame_number:")
            print("frmae_number: ", lq_1_idx)

        # save gt image paths
        gt_rgb_0_path = osp.join(gt_rgb_folder, gt_0_name + '.png')
        gt_rgb_1_path = osp.join(gt_rgb_folder, gt_rgb_name + '.png')
        gt_rgb_2_path = osp.join(gt_rgb_folder, gt_2_name + '.png')
        
        # save moired raw image paths
        lq_raw_0_path = osp.join(lq_raw_folder, lq_0_name + '.npz')
        lq_raw_1_path = osp.join(lq_raw_folder, gt_rgb_name + '.npz')
        lq_raw_2_path = osp.join(lq_raw_folder, lq_2_name + '.npz')
        
        paths.append(dict([
            ('lq_raw_0_path', lq_raw_0_path), 
            ('lq_raw_1_path', lq_raw_1_path), 
            ('lq_raw_2_path', lq_raw_2_path), 
            ('gt_rgb_0_path', gt_rgb_0_path),
            ('gt_rgb_1_path', gt_rgb_1_path),
            ('gt_rgb_2_path', gt_rgb_2_path),
            ('key', gt_rgb_name)]))

    return paths


##################################################################################


######## VDemoire dataloader for sRGB dataset / N(LQ) to N(GT) ##################################
def vdemoire_data_loader_train_n2n(folders):
    assert len(folders) == 2, (
    'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    # input_folder, gt_folder = folders
    lq_rgb_folder, gt_rgb_folder = folders

    gt_rgb_paths = list(scandir(gt_rgb_folder))
    gt_rgb_names = []
    for gt_rgb_path in gt_rgb_paths:
        gt_rgb_name = osp.basename(gt_rgb_path).split('.jpg')[0]
        gt_rgb_names.append(gt_rgb_name)

    paths = []
    for gt_rgb_name in gt_rgb_names:
        gt_rgb_path = osp.join(gt_rgb_folder, gt_rgb_name + '.jpg')
        scene_idx = gt_rgb_name.split('_')[0]
        
        lq_1_idx = int(gt_rgb_name.split('_')[1])
        if lq_1_idx != 0 and lq_1_idx != 59: # when center frame is not the first and last frame
            # frame number: (lq_0_name, lq_1_name, lq_2_name) = (t-1, t, t+1)
            gt_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)
            gt_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)
            
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)
            
        elif lq_1_idx == 0: # when center frame is the first frame
            gt_0_name = gt_rgb_name
            gt_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)
            
            lq_0_name = gt_rgb_name
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)
            
        elif lq_1_idx == 59: # when center frame is the last frame
            gt_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)
            gt_2_name = gt_rgb_name
            
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)
            lq_2_name = gt_rgb_name
            
        else: 
            print("target frame has incorrect frame_number:")
            print("frmae_number: ", lq_1_idx)

        # save gt image paths (clean sRGB)
        gt_rgb_0_path = osp.join(gt_rgb_folder, gt_0_name + '.jpg')
        gt_rgb_1_path = osp.join(gt_rgb_folder, gt_rgb_name + '.jpg')
        gt_rgb_2_path = osp.join(gt_rgb_folder, gt_2_name + '.jpg')
        
        # save lq image paths (moired sRGB with same resolution as GT)
        lq_rgb_0_path = osp.join(lq_rgb_folder, lq_0_name + '.jpg')
        lq_rgb_1_path = osp.join(lq_rgb_folder, gt_rgb_name + '.jpg')
        lq_rgb_2_path = osp.join(lq_rgb_folder, lq_2_name + '.jpg')
        
        paths.append(dict([
            ('lq_0_path', lq_rgb_0_path), 
            ('lq_1_path', lq_rgb_1_path), 
            ('lq_2_path', lq_rgb_2_path), 
            ('gt_0_path', gt_rgb_0_path),
            ('gt_1_path', gt_rgb_1_path),
            ('gt_2_path', gt_rgb_2_path),
            ('key', gt_rgb_name)]))

    return paths
##########################################################################################
######## dataloader fot sRGB dataset (VDemoire) ##################################
def vdemoire_data_loader_test_n2n(folders):
    assert len(folders) == 2, (
    'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    # input_folder, gt_folder = folders
    lq_rgb_folder, gt_rgb_folder = folders

    gt_rgb_paths = list(scandir(gt_rgb_folder))
    gt_rgb_names = []
    for gt_rgb_path in gt_rgb_paths:
        gt_rgb_name = osp.basename(gt_rgb_path).split('.jpg')[0]
        gt_rgb_names.append(gt_rgb_name)

    paths = []
    for gt_rgb_name in gt_rgb_names:
        gt_rgb_path = osp.join(gt_rgb_folder, gt_rgb_name + '.jpg')
        scene_idx = gt_rgb_name.split('_')[0]
        
        lq_1_idx = int(gt_rgb_name.split('_')[1])
        if lq_1_idx != 0 and lq_1_idx != 59: # when center frame is not the first and last frame
            # frame number: (lq_0_name, lq_1_name, lq_2_name) = (t-1, t, t+1)
            gt_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)
            gt_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)
            
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)
            
        elif lq_1_idx == 0: # when center frame is the first frame
            gt_0_name = gt_rgb_name
            gt_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)
            
            lq_0_name = gt_rgb_name
            lq_2_name = scene_idx + '_' + str(lq_1_idx + 1).zfill(5)
            
        elif lq_1_idx == 59: # when center frame is the last frame
            gt_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)
            gt_2_name = gt_rgb_name
            
            lq_0_name = scene_idx + '_' + str(lq_1_idx - 1).zfill(5)
            lq_2_name = gt_rgb_name
            
        else: 
            print("target frame has incorrect frame_number:")
            print("frmae_number: ", lq_1_idx)

        # save gt image paths (clean sRGB)
        gt_rgb_0_path = osp.join(gt_rgb_folder, gt_0_name + '.jpg')
        gt_rgb_1_path = osp.join(gt_rgb_folder, gt_rgb_name + '.jpg')
        gt_rgb_2_path = osp.join(gt_rgb_folder, gt_2_name + '.jpg')
        
        # save lq image paths (moired sRGB with same resolution as GT)
        lq_rgb_0_path = osp.join(lq_rgb_folder, lq_0_name + '.jpg')
        lq_rgb_1_path = osp.join(lq_rgb_folder, gt_rgb_name + '.jpg')
        lq_rgb_2_path = osp.join(lq_rgb_folder, lq_2_name + '.jpg')
        
        paths.append(dict([
            ('lq_0_path', lq_rgb_0_path), 
            ('lq_1_path', lq_rgb_1_path), 
            ('lq_2_path', lq_rgb_2_path), 
            ('gt_0_path', gt_rgb_0_path),
            ('gt_1_path', gt_rgb_1_path),
            ('gt_2_path', gt_rgb_2_path),
            ('key', gt_rgb_name)]))

    return paths

def single_rawrgb_frame_from_folders(folders):
    """
    LQ (raw) 폴더와 GT (rgb) 폴더에서 1:1로 매칭되는 파일 경로 리스트를 생성합니다.
    파일명 규칙은 'v{clip_num}_{frame_num}' 형태입니다. (예: v001_01.npz, v001_01.png)

    Args:
        folders (list): [lq_raw_folder, gt_rgb_folder]

    Returns:
        list[dict]: [{'lq_path': ..., 'gt_path': ..., 'key': ...}, ...]
    """
    assert len(folders) == 2, (
        f'The len of folders should be 2 with [lq_folder, gt_folder]. But got {len(folders)}')
    lq_raw_folder, gt_rgb_folder = folders

    # GT 폴더에서 .png 파일만 수집
    gt_files = sorted(list(scandir(gt_rgb_folder)))
    
    paths = []
    for gt_file in gt_files:
        if not gt_file.endswith('.png'):
            continue
            
        # 파일 경로에서 파일 이름과 확장자 분리
        basename = osp.splitext(osp.basename(gt_file))[0]  # 예: v001_01
        
        # 완전한 경로 구성
        gt_path = osp.join(gt_rgb_folder, gt_file)
        lq_path = osp.join(lq_raw_folder, f'{basename}.npz')
        
        # 두 파일 모두 존재하는지 확인
        if osp.exists(lq_path) and osp.exists(gt_path):
            paths.append(dict([
                ('lq_path', lq_path),
                ('gt_path', gt_path),
                ('key', basename),
            ]))
        else:
            missing = []
            if not osp.exists(lq_path):
                missing.append(f"LQ: {lq_path}")
            if not osp.exists(gt_path):
                missing.append(f"GT: {gt_path}")
            print(f"Warning: Missing files for {basename}: {', '.join(missing)}")
            
    print(f"Found {len(paths)} pairs of GT and LQ files")
    return paths
##########################################################################################


def tensor2numpy(tensor):
    img_np = tensor.squeeze().numpy()
    img_np[img_np < 0] = 0
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    return img_np.astype(np.float32)


def imwrite_gt(img, img_path, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(img_path))
        os.makedirs(dir_name, exist_ok=True)

    img = img.clip(0, 1.0)
    uint8_image = np.round(img * 255.0).astype(np.uint8)
    cv2.imwrite(img_path, uint8_image)
    return None


def read_img(img_path):
    img = cv2.imread(img_path, -1)
    if img is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {img_path}. 파일이 존재하는지 확인하세요.")
    return img / 255.

def read_npz(path):
    raw_img = np.load(path)
    raw_data = raw_img['data'].transpose((2,0,1))
    bl = raw_img['black_level_per_channel'][0]
    wl = raw_img['white_level']
    norm_factor = wl - bl
    raw_data = (raw_data- bl)/norm_factor
    raw_data = raw_data.astype(np.float32)
    # add camera_whitebalance
    cwb = raw_img['camera_whitebalance']
    cwb_rggb = np.expand_dims(np.expand_dims(np.array([cwb[0],cwb[1],cwb[1],cwb[2]]), axis=1), axis=2)
    raw_data = raw_data*cwb_rggb
    raw_data = raw_data.astype(np.float32)
    raw_data = raw_data.transpose((1,2,0))
    return raw_data

def read_npz_16bit(path):
    """
    메타데이터 없이 16비트(2^16) RAW 데이터를 불러와서 normalize합니다.
    
    Args:
        path (str): NPZ 파일 경로
        
    Returns:
        np.ndarray: 정규화된 RAW 이미지 데이터 (0~1 범위, float32 타입, HWC 형식)
    """
    # NPZ 파일 로드
    try:
        raw_img = np.load(path)
        
        # 'data' 키가 있는 경우
        if 'data' in raw_img:
            raw_data = raw_img['data']
        else:
            # 첫 번째 항목을 데이터로 간주
            raw_data = raw_img[list(raw_img.keys())[0]]
        
        # 채널이 마지막 차원에 있지 않은 경우 전치
        if raw_data.shape[0] == 4 and len(raw_data.shape) == 3:
            raw_data = raw_data.transpose((1, 2, 0))  # CHW -> HWC
            
        # 16비트(65535) 기준으로 정규화
        max_value = 65535.0  # 2^16 - 1
        raw_data = raw_data.astype(np.float32) / max_value
        
        return raw_data
        
    except Exception as e:
        print(f"NPZ 파일 로드 중 오류 발생: {e}")
        raise