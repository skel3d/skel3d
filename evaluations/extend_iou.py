import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

import csv
import os


def compute_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    #if boxAArea > boxBArea:
    #    iou = 1 + (1-iou)
    
    return iou

def get_bounding_box(image):
    # Convert the image to grayscale and threshold it
    # Find the pixels that are not white

    non_white_pixels = image != 255
    non_white_pixels = np.where(non_white_pixels)


    if len(non_white_pixels)==1 or len(non_white_pixels[0]) == 0:
        return [0, 0, 0, 0]

    
    # Get the bounding box coordinates of the non-white pixels
    x = np.min(non_white_pixels[1])
    y = np.min(non_white_pixels[0])
    w = np.max(non_white_pixels[1]) - x
    h = np.max(non_white_pixels[0]) - y

    # Return the bounding box
    return [x, y, x + w, y + h]

def skeleton_coverage(silhouette, skeleton):
    # Ensure binary images
    silhouette = silhouette < 255
    skeleton = skeleton < 255


    if np.sum(skeleton) == 0:
        return 0, 0, 0
    
    
    # Calculate coverage ratio
    coverage_ratio = np.sum(skeleton & silhouette) / np.sum(skeleton)
    
    # Distance transform
    distance_map = distance_transform_edt(silhouette)
    skeleton_distances = distance_map[skeleton]

    
    
    # Average and maximum distances
    avg_distance = np.mean(skeleton_distances)
    max_distance = np.max(skeleton_distances)
    
    return coverage_ratio, avg_distance, max_distance


file_in = 'skel3d.csv'

file_out = 'skel3d_ext.csv'
#file2 = 'results_base.csv'

target_path = '/outputs/hdf5_skel3d_warp_epoch=000019.ckpt/inputs/'
skel_path = '/outputs/hdf5_skel3d_warp_epoch=000019.ckpt/timegt/'



ious = []
score_differences = []

coverage_ratios = []
avg_distances = []
max_distances = []


out = []

with open(file_in, 'r') as f1:
        reader1 = csv.DictReader(f1)
        for row1 in  reader1:
                
                
                 
                
                filename = row1['id']
                target = cv2.imread(target_path + filename, cv2.IMREAD_GRAYSCALE)
                skel = cv2.imread(skel_path + filename, cv2.IMREAD_GRAYSCALE)
                
                # Get the bounding box of the skeleton image
                box_skel = get_bounding_box(skel)
                
                # Get the bounding box of the target image
                box_target = get_bounding_box(target)
                
                
                # Compute the intersection over union of the bounding boxes
                iou1 = compute_iou(box_skel, box_target)

                row1['iou'] = iou1

                out.append(row1)

    
with open(file_out, 'w') as f2:
        writer = csv.DictWriter(f2, fieldnames=out[0].keys())
        writer.writeheader()
        for row in out:
            writer.writerow(row)

print('Done')