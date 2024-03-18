from prompt_points import PromptGeneration, SpecialPointItem, CrossPointItem
from torch.utils.data import Dataset
import cv2
import os
from scipy.stats import multivariate_normal
import numpy as np
from scipy import ndimage
from collections import *
from itertools import product
from display import show_result_sample_figure
import albumentations as alb
from tqdm import tqdm


class SpecialPoint_Dataset(Dataset):
    def __init__(self, 
                 image_size=(1024, 1024),
                 point_type="Endpoint", # Bifurcation, Endpoint
                 label_type="LargeVessel",
                 is_training=True):
        self.image_size = image_size
        self.is_training = is_training

        self.samples, self.heatmaps, self.sample_ids = [], [], []

        w, h = image_size

        # Bifurcation
        point_dir = "datasets/{}".format(point_type)
        intermediate_dir = "intermediate/{}".format(point_type)
        if not os.path.exists(intermediate_dir): os.makedirs(intermediate_dir)

        annotated_dct = {}
        for file_name in tqdm(os.listdir(point_dir), desc=point_type):
            sample_id = file_name[:-4]
            npy_file_path = "{}/{}.npy".format(intermediate_dir, sample_id)

            if os.path.exists(npy_file_path):
                heatmaps = np.load(npy_file_path)
            else:
                with open("{}/{}".format(point_dir, file_name), 'r') as file:
                    lines = file.readlines()
                centers = []
                for line in lines[1:]:
                    x, y = map(float, line[:-1].split())
                    centers.append((int(h * y), int(w * x)))
                heatmaps = self.points_to_gaussian_heatmap(centers)
                np.save(npy_file_path, heatmaps)
            annotated_dct[sample_id] = heatmaps
        
        for fov in "3M", "6M":
            label_dir = "datasets/OCTA-500/OCTA_{}/GT_{}".format(fov, label_type)
            for sample_path in sorted(os.listdir(label_dir)):
                sample_id = sample_path[:-4]

                sample = cv2.imread("{}/{}.bmp".format(label_dir, sample_id), cv2.IMREAD_GRAYSCALE) / 255
                sample = cv2.resize(sample, image_size)

                if sample_id in annotated_dct and is_training:
                    self.samples.append(sample)
                    self.heatmaps.append(annotated_dct[sample_id])
                    self.sample_ids.append(sample_id)
                
                if not is_training:
                    self.samples.append(sample)
                    self.heatmaps.append(np.zeros_like(sample))
                    self.sample_ids.append(sample_id)

        self.num_of_samples = len(self.samples)

        prob = 0.5
        self.transform = alb.Compose([
            alb.SafeRotate(limit=45, p=prob),
            alb.VerticalFlip(p=prob),
            alb.HorizontalFlip(p=prob),
        ])
        
    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, index):
        sample, heatmap, sample_id = self.samples[index], self.heatmaps[index], self.sample_ids[index]

        if self.is_training:
            transformed = self.transform(**{"image": sample, "mask": heatmap})
            sample, heatmap = transformed["image"], transformed["mask"]
        
        return sample[np.newaxis,:], heatmap[np.newaxis,:], sample_id

    def points_to_gaussian_heatmap(self, centers, scale=32): 
        width, height = self.image_size
        gaussians = []
        for y, x in centers:
            s = np.eye(2) * scale
            g = multivariate_normal(mean=(x, y), cov=s)
            gaussians.append(g)
        x, y = np.arange(0, width), np.arange(0, height)
        xx, yy = np.meshgrid(x, y)
        xxyy = np.stack([xx.ravel(), yy.ravel()]).T
        zz = sum(g.pdf(xxyy) for g in gaussians)
        img = zz.reshape((height, width))
        return img / np.max(img)
    
class Octa500_Dataset(Dataset):
    def __init__(
            self, 
            fov="3M", 
            label_type="Artery", 
            prompt_positive_num=1, 
            prompt_negative_num=1, 
            is_local=True,
            is_training=True,
            random_seed=0,
            ):
    
        self.is_training = is_training
        
        modal = "OCTA"
        layers = ["OPL_BM", "ILM_OPL", "FULL"]
        data_dir = "datasets/OCTA-500"
        label_dir = "{}/OCTA_{}/GT_{}".format(data_dir, fov, label_type)
        self.sample_ids = [x[:-4] for x in sorted(os.listdir(label_dir))]
        
        images = []
        for sample_id in self.sample_ids:
            image_channels = []
            for layer in layers:
                image_path = "{}/OCTA_{}/ProjectionMaps/{}({})/{}.bmp".format(data_dir, fov, modal, layer, sample_id)
                image_channels.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
            images.append(np.array(image_channels))
        self.images = images

        load_label = lambda sample_id: cv2.imread("{}/{}.bmp".format(label_dir, sample_id), cv2.IMREAD_GRAYSCALE) / 255
        self.labels = [load_label(x) for x in self.sample_ids]

        prob = 0.3
        self.transform = alb.Compose([
            alb.RandomBrightnessContrast(p=prob),
            alb.CLAHE(p=prob), 
            # alb.SafeRotate(limit=15, p=prob),
            alb.VerticalFlip(p=prob),
            alb.HorizontalFlip(p=prob),
            alb.AdvancedBlur(p=prob),
        ])

        self.pg = PromptGeneration(
            random_seed=random_seed * int(1-is_training), 
            neg_range=(0, 9), # general: 0-9 
            positive_num=prompt_positive_num, 
            negative_num=prompt_negative_num,
            is_local=is_local,
            label_type=label_type,
        )

        self.sam_items = []
        self.num_of_samples = len(self.images)

        self.sample_counter = Counter()

        sample_id2subset = {}
        range_items = [(10001, 10181, "training"), (10181, 10201, "validation"), (10201, 10301, "test")] # 6M
        range_items += [(10301, 10441, "training"), (10441, 10451, "validation"), (10451, 10501, "test")] # 3M

        for l, r, subset in range_items:
            for sample_id in range(l, r): sample_id2subset[str(sample_id)] = subset

        if not is_training:
            if is_local:
                for index in tqdm(range(len(self.labels)), desc="loading_val_data"):
                    cid = 0
                    for selected_component, prompt_points_pos, prompt_points_neg in \
                        self.pg.label_to_all_local_components(self.labels[index], self.sample_ids[index]):
                        prompt_type = np.array([1] * len(prompt_points_pos) + [0] * len(prompt_points_neg))
                        prompt_points = np.array(prompt_points_pos + prompt_points_neg)
                        sample_id = "{}-{:0>2}".format(self.sample_ids[index], cid)
                        self.sam_items.append((self.images[index], prompt_points, prompt_type, selected_component, sample_id))
                        cid += 1
                    self.sample_counter[sample_id2subset[self.sample_ids[index]]] += cid
            else:
                for index in tqdm(range(len(self.labels)), desc="loading_val_data"):
                    selected_component, prompt_points_pos, prompt_points_neg = \
                        self.pg.get_prompt_point(self.labels[index], self.sample_ids[index])
                    prompt_type = np.array([1] * len(prompt_points_pos) + [0] * len(prompt_points_neg))
                    prompt_points = np.array(prompt_points_pos + prompt_points_neg)
                    self.sam_items.append((self.images[index], prompt_points, prompt_type, selected_component, self.sample_ids[index]))
            self.num_of_samples = len(self.sam_items)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, index):
        if self.is_training:
            image, label, sample_id = self.images[index], self.labels[index], self.sample_ids[index]
            transformed = self.transform(**{"image": image.transpose((1,2,0)), "mask": label[np.newaxis,:].transpose((1,2,0))})
            image, label = transformed["image"].transpose((2,0,1)), transformed["mask"].transpose((2,0,1))[0]

            selected_component, prompt_points_pos, prompt_points_neg = self.pg.get_prompt_point(label, sample_id)
            prompt_type = np.array([1] * len(prompt_points_pos) + [0] * len(prompt_points_neg))
            prompt_points = np.array(prompt_points_pos + prompt_points_neg)
            return image, prompt_points, prompt_type, selected_component, sample_id
        return self.sam_items[index]


class Octa500_Dataset_SpecialPoints(Dataset):
    def __init__(
            self, 
            fov="3M", 
            label_type="Artery", 
            is_local=True,
            point_type="Endpoint", 
            is_training=True,
            random_seed=0,
            ):
    
        self.is_training = is_training
        
        modal = "OCTA"
        layers = ["OPL_BM", "ILM_OPL", "FULL"]
        data_dir = "datasets/OCTA-500"
        label_dir = "{}/OCTA_{}/GT_{}".format(data_dir, fov, label_type)
        self.sample_ids = [x[:-4] for x in sorted(os.listdir(label_dir))]
        
        images = []
        for sample_id in self.sample_ids:
            image_channels = []
            for layer in layers:
                image_path = "{}/OCTA_{}/ProjectionMaps/{}({})/{}.bmp".format(data_dir, fov, modal, layer, sample_id)
                image_channels.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
            images.append(np.array(image_channels))
        self.images = images

        load_label = lambda sample_id: cv2.imread("{}/{}.bmp".format(label_dir, sample_id), cv2.IMREAD_GRAYSCALE) / 255
        self.labels = [load_label(x) for x in self.sample_ids]

        prob = 0.3
        self.transform = alb.Compose([
            alb.RandomBrightnessContrast(p=prob),
            alb.CLAHE(p=prob), 
            # alb.SafeRotate(limit=15, p=prob),
            alb.VerticalFlip(p=prob),
            alb.HorizontalFlip(p=prob),
            alb.AdvancedBlur(p=prob),
        ])

        self.spi = SpecialPointItem(
            fov=fov, 
            label_type=label_type,
            point_type=point_type, 
            is_local=is_local,
            random_seed=random_seed
        )


        self.sam_items = []
        self.num_of_samples = len(self.images)

        self.sample_counter = Counter()

        sample_id2subset = {}
        range_items = [(10001, 10181, "training"), (10181, 10201, "validation"), (10201, 10301, "test")] # 6M
        range_items += [(10301, 10441, "training"), (10441, 10451, "validation"), (10451, 10501, "test")] # 3M

        for l, r, subset in range_items:
            for sample_id in range(l, r): sample_id2subset[str(sample_id)] = subset


        cnt = 0

        cnt2 = 0

        if not is_training:
            for index in tqdm(range(len(self.labels)), desc="loading_val_data"):
                cid = 0
                for prompt_points, prompt_type, selected_component in self.spi.get_items(self.sample_ids[index]):
                    sample_id = "{}-{:0>2}".format(self.sample_ids[index], cid)
                    self.sam_items.append((self.images[index], prompt_points, prompt_type, selected_component, sample_id))

                    cnt += len(prompt_type)
                    cnt2 += 1

                    cid += 1
                self.sample_counter[sample_id2subset[self.sample_ids[index]]] += cid
            self.num_of_samples = len(self.sam_items)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, index):
        if self.is_training:
            image, label, sample_id = self.images[index], self.labels[index], self.sample_ids[index]

            prompt_points, prompt_type, selected_component = self.spi.get_single_item(self.sample_ids[index]) 
            transformed = self.transform(**{"image": image.transpose((1,2,0)), "mask": label[np.newaxis,:].transpose((1,2,0))})
            image, label = transformed["image"].transpose((2,0,1)), transformed["mask"].transpose((2,0,1))[0]

            return image, prompt_points, prompt_type, selected_component, sample_id
        return self.sam_items[index]
    
# cross
class Octa500_Dataset_Cross(Dataset):
    def __init__(
        self, 
        fov="3M", 
        label_type="Artery", 
        is_local=False,
        is_training=True,
        random_seed=0,
        ):

        self.is_training = is_training
        
        modal = "OCTA"
        layers = ["OPL_BM", "ILM_OPL", "FULL"]
        data_dir = "datasets/OCTA-500"
        label_dir = "{}/OCTA_{}/GT_{}".format(data_dir, fov, label_type)
        self.sample_ids = [x[:-4] for x in sorted(os.listdir(label_dir))]

        images = []
        for sample_id in self.sample_ids:
            image_channels = []
            for layer in layers:
                image_path = "{}/OCTA_{}/ProjectionMaps/{}({})/{}.bmp".format(data_dir, fov, modal, layer, sample_id)
                image_channels.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
            images.append(np.array(image_channels))
        self.images = images

        load_label = lambda sample_id: cv2.imread("{}/{}.bmp".format(label_dir, sample_id), cv2.IMREAD_GRAYSCALE) / 255
        self.labels = [load_label(x) for x in self.sample_ids]

        prob = 0.3
        self.transform = alb.Compose([
            alb.RandomBrightnessContrast(p=prob),
            alb.CLAHE(p=prob), 
            # alb.SafeRotate(limit=15, p=prob),
            alb.VerticalFlip(p=prob),
            alb.HorizontalFlip(p=prob),
            alb.AdvancedBlur(p=prob),
        ])

        self.cpi = CrossPointItem(
            fov=fov, 
            label_type=label_type,
            is_local=is_local,
            random_seed=random_seed
        )

        self.sam_items = []
        self.num_of_samples = len(self.images)

        self.sample_counter = Counter()

        sample_id2subset = {}
        range_items = [(10001, 10181, "training"), (10181, 10201, "validation"), (10201, 10301, "test")] # 6M
        range_items += [(10301, 10441, "training"), (10441, 10451, "validation"), (10451, 10501, "test")] # 3M

        for l, r, subset in range_items:
            for sample_id in range(l, r): sample_id2subset[str(sample_id)] = subset

        if not is_training:
            for index in tqdm(range(len(self.labels)), desc="loading_val_data"):
                sample_id = self.sample_ids[index]
                prompt_points, prompt_type, selected_component = self.cpi.get_single_item(sample_id)
                self.sam_items.append((self.images[index], prompt_points, prompt_type, selected_component, sample_id))
                self.sample_counter[sample_id2subset[self.sample_ids[index]]] += 1
            self.num_of_samples = len(self.sam_items)
        
    
    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, index):
        if self.is_training:
            image, label, sample_id = self.images[index], self.labels[index], self.sample_ids[index]

            prompt_points, prompt_type, selected_component = self.cpi.get_single_item(self.sample_ids[index]) 
            transformed = self.transform(**{"image": image.transpose((1,2,0)), "mask": label[np.newaxis,:].transpose((1,2,0))})
            image, label = transformed["image"].transpose((2,0,1)), transformed["mask"].transpose((2,0,1))[0]

            return image, prompt_points, prompt_type, selected_component, sample_id
        return self.sam_items[index]
