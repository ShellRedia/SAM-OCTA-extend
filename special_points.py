import os
import cv2
import numpy
from scipy import ndimage
import numpy as np
from tqdm import tqdm


def get_labels():
    dir_3M = "datasets/OCTA-500/OCTA_3M/GT_LargeVessel"
    dir_6M = "datasets/OCTA-500/OCTA_6M/GT_LargeVessel"

    label_dct = {}

    for sample_path in sorted(os.listdir(dir_6M)):
        label = cv2.imread("{}/{}".format(dir_6M, sample_path), cv2.IMREAD_COLOR)
        label_dct[sample_path[:-4]] = label

    for sample_path in sorted(os.listdir(dir_3M)):
        label = cv2.imread("{}/{}".format(dir_3M, sample_path), cv2.IMREAD_COLOR)
        label_dct[sample_path[:-4]] = label
    
    return label_dct

label_dct = get_labels()
ends_point_dir = "datasets/Endpoint"
for sample_path in os.listdir(ends_point_dir):
    sample_id = sample_path[:-4]
    label = label_dct[sample_id]
    w, h, c = label.shape
    with open("{}/{}".format(ends_point_dir, sample_path), 'r') as file:
        lines = file.readlines()
    coords = []
    for line in lines[1:]:
        x, y = map(float, line[:-1].split())
        x, y = int(w * x), int(h * y)
        cv2.circle(label, (x, y), 4, (0, 0, 255), -1)
        coords.append([y, x])
    fov = "6M" if sample_id <= "10300" else "3M"
    w0 = {"3M":304, "6M":400}[fov]
    np.save("datasets/OCTA-500/OCTA_{}/GT_Point/Endpoint/{}.npy".format(fov, sample_id), coords)
    cv2.imwrite("special_points/{}.png".format(sample_id), label)
            

structure = ndimage.generate_binary_structure(2, 2)

to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)
overlay = lambda x, y: cv2.addWeighted(x, 0.5, y, 0.5, 0)


pred_dir = "special_points/Endpoints/2024-01-27-01-16-25/500"


for image_path in tqdm(sorted(os.listdir(pred_dir))):
    sample_id = image_path[:-4]
    fov = "6M" if sample_id <= "10300" else "3M"
    w0 = {"3M":304, "6M":400}[fov]
    
    label = cv2.imread("{}/{}".format(pred_dir, image_path), cv2.IMREAD_GRAYSCALE)
    h, w = label.shape

    scaling = lambda x: int(x / h * w0)
    label, heatmap = label[:, :w // 2], label[:, w // 2:]
    labelmaps, connected_num = ndimage.label(heatmap, structure=structure)
    coords = []
    for pixel_val in range(1, connected_num+1):
        indices = np.where(labelmaps == pixel_val)
        end_point = tuple(map(scaling, [np.mean(indices[0]), np.mean(indices[1])]))
        coords.append(end_point)
    np.save("datasets/OCTA-500/OCTA_{}/GT_Point/Endpoint/{}.npy".format(fov, sample_id), coords)
    print(coords)
    label = to_3ch(label)
    label = overlay(label, label)
    for x, y in coords: cv2.circle(label, (y, x), 4, (0, 255, 0), -1)
    cv2.imwrite("temp.png", label)
    break
image = cv2.imread("datasets/OCTA-500/OCTA_6M/ProjectionMaps/OCTA(OPL_BM)/10001.bmp", cv2.IMREAD_COLOR)
coords = np.load("datasets/OCTA-500/OCTA_6M/GT_Point/Endpoint/10001.npy")
# print(list(coords))
for x, y in coords:
    cv2.circle(image, (y, x), 4, (0, 255, 0), -1)
cv2.imwrite("temp.png", image)