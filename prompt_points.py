import numpy as np
from scipy import ndimage
import cv2, os, random
from collections import *
from itertools import *
from functools import *
from display import show_prompt_points_image
from tqdm import tqdm

class PromptGeneration:
    def __init__(self, 
                 random_seed=0, 
                 neg_range=(6, 9),
                 negative_method="dfs",
                 positive_num=2, 
                 negative_num=2,
                 is_local=True,
                 label_type="LargeVessel",
                 is_specific=False):
        
        if random_seed:  
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.random_seed = random_seed
        self.neg_range = neg_range
        self.positive_num = positive_num
        self.negative_num = negative_num
        self.is_local = is_local
        self.label_type = label_type
        self.is_specific = is_specific
        
        profile_str = "-".join(map(str, [neg_range, is_local, label_type, is_specific]))
        self.cache_dir = "intermediate/{}".format(profile_str)

        print("prompt points cache_dir:", self.cache_dir)
        if not os.path.exists(self.cache_dir): os.makedirs(self.cache_dir)

        self.min_area = 20
        self.const_point = (-100, -100)
        self.search_negative_region = {
            "dfs":self.search_negative_region_dfs,
            "numpy":self.search_negative_region_numpy
        }[negative_method]

    def get_labelmap(self, label):
        # print("label:", label.max()) # 255
        structure = ndimage.generate_binary_structure(2, 2)
        labelmaps, connected_num = ndimage.label(label, structure=structure)

        label = np.zeros_like(labelmaps)
        for i in range(1, 1+connected_num):
            if np.sum(labelmaps==i) >= self.min_area: label += np.where(labelmaps==i, 255, 0)

        structure = ndimage.generate_binary_structure(2, 2)
        labelmaps, connected_num = ndimage.label(label, structure=structure)

        return labelmaps, connected_num

    def search_negative_region_dfs(self, labelmap):
        def search(neg_range):
            w, h = labelmap.shape 
            dq = deque([])
            connected_points_pos_cnt = Counter()
            for (x, y), val in np.ndenumerate(labelmap):
                if val: connected_points_pos_cnt[val] += 1
            for (x, y), val in np.ndenumerate(labelmap):
                if val and connected_points_pos_cnt[val] >= self.min_area: dq.append((val, x, y, 0))
            negative_region = labelmap.copy().astype(np.int32)
            while dq:
                val, x, y, step = dq.popleft()
                if step >= neg_range: continue
                for dx, dy in product(*[range(-1, 2)] * 2):
                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h: continue
                    if negative_region[(nx, ny)] == 0: 
                        negative_region[(nx, ny)] = val
                        dq.append((val, nx, ny, step+1))
                    else:
                        if negative_region[(nx, ny)] in (-1, val): continue
                        else: negative_region[(nx, ny)] = -1
            negative_region -= labelmap
            return np.maximum(negative_region, 0).astype(np.uint8)
        inner_range, outer_range = self.neg_range
        return search(outer_range) - search(inner_range)
    
    def search_negative_region_numpy(self, labelmap):
        inner_range, outer_range = self.neg_range
        def search(neg_range):
            kernel = np.ones((neg_range * 2 + 1, neg_range * 2 + 1), np.uint8)
            negative_region = cv2.dilate(labelmap, kernel, iterations=1)
            mx = labelmap.max() + 1
            labelmap_r = (mx - labelmap) * np.minimum(1, labelmap)
            r = cv2.dilate(labelmap_r, kernel, iterations=1)
            negative_region_r = (r.astype(np.int32) - mx) * np.minimum(1, r)
            diff = negative_region.astype(np.int32) + negative_region_r
            overlap = np.minimum(1, np.abs(diff).astype(np.uint8))
            return negative_region - overlap - labelmap
        return search(outer_range) - search(inner_range) 
    
    def get_prompt_region(self, label, key=""):
        # "FAZ", "Capillary"
        def calculate_regions():
            if self.is_specific:
                specific_region_func = {
                    "FAZ" : self.get_specific_FAZ,
                    "Capillary" : self.get_specific_Capillary,
                }[self.label_type]
                labelmaps, negative_region, connected_num = specific_region_func(label)
            else:
                labelmaps, connected_num = self.get_labelmap(label)
                negative_region = self.search_negative_region(labelmaps)
            return labelmaps, negative_region, connected_num

        if key:
            positive_region_file_path = "{}/{}_p.npy".format(self.cache_dir, key)
            negative_region_file_path = "{}/{}_n.npy".format(self.cache_dir, key)
            if os.path.exists(negative_region_file_path) and os.path.exists(positive_region_file_path): 
                labelmaps = np.load(positive_region_file_path)
                negative_region = np.load(negative_region_file_path)
                connected_num = labelmaps.max()
            else:
                labelmaps, negative_region, connected_num = calculate_regions()
                np.save(positive_region_file_path, labelmaps)
                np.save(negative_region_file_path, negative_region)
        else:
            labelmaps, negative_region, connected_num = calculate_regions()
        return labelmaps, negative_region, connected_num

    def label_to_point_prompt_global(self, label, key=""):
        positive_points, negative_points = [], []
        connected_points_pos, connected_points_neg = defaultdict(list), defaultdict(list)

        labelmaps, negative_region, connected_num = self.get_prompt_region(label, key)

        selected_labelmap = label

        for (x, y), val in np.ndenumerate(labelmaps): connected_points_pos[val].append((y, x))
        for (x, y), val in np.ndenumerate(negative_region): connected_points_neg[val].append((y, x))
        
        for connected_id in range(1, connected_num+1):
            len_p, len_n = len(connected_points_pos[connected_id]), len(connected_points_neg[connected_id])
            if min(len_p, len_n) >= self.min_area:
                positive_points += random.sample(connected_points_pos[connected_id], self.positive_num)
                negative_points += random.sample(connected_points_neg[connected_id], self.negative_num)
            else:
                for y, x in connected_points_pos[connected_id]: selected_labelmap[(x, y)] = 0
                for y, x in connected_points_neg[connected_id]: negative_region[(x, y)] = 0

        if not (positive_points + negative_points): negative_points = [self.const_point]

        return np.array([selected_labelmap], dtype=float), negative_region, positive_points, negative_points

    def label_to_point_prompt_local(self, label, key=""):
        labelmaps, negative_region, connected_num = self.get_prompt_region(label, key)
        labelmap_points = []

        gp, gn = defaultdict(list), defaultdict(list)

        for (x, y), val in np.ndenumerate(labelmaps):
            if val: gp[val].append((x, y))
        for (x, y), val in np.ndenumerate(negative_region):
            if val: gn[val].append((x, y))

        for k in gp:
            if min(len(gp[k]), len(gn[k])) >= self.min_area:
                labelmap_points += gp[k]
        
        selected_pixel = random.randint(0, len(labelmap_points)-1)
        selected_val = labelmaps[labelmap_points[selected_pixel]]

        selected_labelmap = np.where(labelmaps == selected_val, labelmaps, 0) // selected_val
        negative_region = np.where(negative_region == selected_val, negative_region, 0) // selected_val

        positive_points = [(y, x) for (x, y), val in np.ndenumerate(selected_labelmap) if val]
        negative_points = [(y, x) for (x, y), val in np.ndenumerate(negative_region) if val]

        positive_points = random.sample(positive_points, self.positive_num)
        negative_points = random.sample(negative_points, self.negative_num)

        if not (positive_points + negative_points): negative_points = [self.const_point]

        return np.array([selected_labelmap], dtype=float), negative_region, positive_points, negative_points
    
    def label_to_all_local_components(self, label, key=""):
        labelmaps, negative_region, connected_num = self.get_prompt_region(label, key)
        negative_region = self.search_negative_region(labelmaps.astype(np.uint8))

        gp, gn = defaultdict(list), defaultdict(list)

        for (x, y), val in np.ndenumerate(labelmaps): gp[val].append((y, x))
        for (x, y), val in np.ndenumerate(negative_region): gn[val].append((y, x))

        components_array = []
        for i in range(1, connected_num+1):
            if min(len(gp[i]), len(gn[i])) >= self.min_area:
                ppn = random.sample(gp[i], self.positive_num)
                pnn = random.sample(gn[i], self.negative_num)
                selected_labelmap = np.where(labelmaps == i, labelmaps, 0) // i
                if not (ppn + pnn): pnn = [self.const_point]
                components_array.append((np.array([selected_labelmap], dtype=float), ppn, pnn))

        return components_array
    
    def get_specific_FAZ(self, label):
        inner_range, outer_range = self.neg_range
        kernel = np.ones((outer_range * 2 + 1, outer_range * 2 + 1), np.uint8)
        negative_region = cv2.dilate(label, kernel, iterations=1) - label

        outline = cv2.dilate(label, np.ones((3, 3), np.uint8), iterations=1) - label
        kernel = np.ones((outer_range * 2 - 1, outer_range * 2 - 1), np.uint8)
        positive_region = cv2.dilate(outline, kernel, iterations=1) - negative_region

        return positive_region, negative_region, positive_region.max()
    
    def get_specific_Capillary(self, label):
        positive_region = np.minimum(1, label)
        negative_region = self.search_negative_region(positive_region)
        return positive_region, negative_region, positive_region.max()
    
    def get_prompt_point(self, label, key=""):
        selected_component, negative_region, prompt_points_pos, prompt_points_neg = \
            self.label_to_point_prompt_local(label, key) if self.is_local else self.label_to_point_prompt_global(label, key)
        return selected_component, prompt_points_pos, prompt_points_neg
    
    # for debug ...
    def draw_generated_label(self, image, label, key="", save_file="temp.png"):
        selected_component, negative_region, prompt_points_pos, prompt_points_neg = \
            self.label_to_point_prompt_local(label, key) if self.is_local else self.label_to_point_prompt_global(label, key)

        alpha = 0.5
        overlay = lambda x, y: cv2.addWeighted(x, alpha, y, 1-alpha, 0)

        to_red = lambda x: np.array([np.zeros_like(x), np.zeros_like(x), x]).transpose((1,2,0)).astype(dtype=np.uint8)
        to_yellow = lambda x: np.array([np.zeros_like(x), x, x]).transpose((1,2,0)).astype(dtype=np.uint8)

        processed_image = to_yellow(selected_component[0]) + to_red(negative_region)
        processed_image = overlay(image, processed_image)

        for x, y in prompt_points_pos:
            cv2.circle(processed_image, (x, y), 5, (255, 0, 0), -1)
            cv2.circle(processed_image, (x, y), 4, (0, 255, 0), -1)
        for x, y in prompt_points_neg:
            cv2.circle(processed_image, (x, y), 5, (255, 0, 0), -1)
            cv2.circle(processed_image, (x, y), 4, (0, 0, 255), -1)

        cv2.imwrite(save_file, processed_image)       
        

class SpecialPointItem:
    def __init__(self, 
                 fov="3M", 
                 label_type="LargeVessel",
                 point_type="Endpoint", 
                 is_local=True,
                 random_seed=0):

        if random_seed:  
            random.seed(random_seed)
            np.random.seed(random_seed)
            
        self.sam_items = {}

        self.min_area = 20


        structure = ndimage.generate_binary_structure(2, 2)

        label_dir = "datasets/OCTA-500/OCTA_{}/GT_{}".format(fov, label_type)
        point_dir = "datasets/OCTA-500/OCTA_{}/GT_Point/{}/{}".format(fov, label_type, point_type)

        sample_ids = [x[:-4] for x in sorted(os.listdir(label_dir))]

        for sample_id in tqdm(sample_ids):
            label = cv2.imread("{}/{}.bmp".format(label_dir, sample_id), cv2.IMREAD_GRAYSCALE)
            h, w = label.shape
            coords = np.load("{}/{}.npy".format(point_dir, sample_id))

            if not is_local:
                self.sam_items[sample_id] = [(coords, len(coords) * [1], label)]
            else:
                g = defaultdict(list)
                labelmap, connected_num = ndimage.label(label, structure=structure)
                
                for sx, sy in coords:
                    mat = np.copy(labelmap)
                    dq = deque([(sx, sy)])
                    vis = set([(sx, sy)])
                    while dq:
                        x, y = dq.popleft()
                        if mat[(x, y)]:
                            if abs(x-sx) + abs(y-sy) <= 5:
                                g[mat[(x, y)]].append([sy, sx])
                            break
                        for dx, dy in product(*[range(-1, 2)] * 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < w and 0 <= ny < w and (nx, ny) not in vis:
                                dq.append((nx, ny))
                                vis.add((nx, ny))
                item_lst = []
                for i in range(1, connected_num+1):
                    sc = np.where(labelmap==i, 1, 0)
                    coord_lst = list(zip(*np.where(sc == 1)))
                    if len(coord_lst) > self.min_area:
                        pp = g[i]
                        item_lst.append((pp, len(pp) * [1], sc))
                self.sam_items[sample_id] = item_lst

    def get_items(self, sample_id):
        item_lst = []
        for prompt_points, prompt_types, select_labelmap in self.sam_items[sample_id]:
            coord_lst = list(zip(*np.where(select_labelmap == 1)))
            if len(coord_lst) > self.min_area:
                if not prompt_points:
                    random_coordinate = random.choice(coord_lst)
                    y, x = random_coordinate
                    prompt_points = [[x, y]]
                    prompt_types = [1]
                item_lst.append((np.array(prompt_points), np.array(prompt_types), np.array([select_labelmap], dtype=float)))

        return item_lst
    
    def get_single_item(self, sample_id):
        flag = True
        while flag:
            prompt_points, prompt_types, select_labelmap = random.choice(self.sam_items[sample_id])
            coord_lst = list(zip(*np.where(select_labelmap == 1)))
            if len(coord_lst) > self.min_area: flag = False

        if not prompt_points:
            random_coordinate = random.choice(coord_lst)
            y, x = random_coordinate
            prompt_points = [[x, y]]
            prompt_types = [1]
        return np.array(prompt_points), np.array(prompt_types), np.array([select_labelmap], dtype=float)

class CrossPointItem:
    def __init__(self, 
                fov="3M", 
                label_type="Artery",
                is_local=True,
                random_seed=0):

        if random_seed:  
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.av_dir = "datasets/OCTA-500/OCTA_{}/GT_AV".format(fov)
        self.is_local = is_local
        self.label_type = label_type

        self.min_area = 20

    def get_single_item(self, sample_id):
        structure = ndimage.generate_binary_structure(2, 2)

        # Global
        prompt_points, prompt_types, select_labelmap = [], [], []

        if not self.is_local:
            label_path = "{}/{}.bmp".format(self.av_dir, sample_id)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            artery = np.where(label == 1, 255, 0) + np.where(label == 3, 255, 0)
            vein = np.where(label == 2, 255, 0) + np.where(label == 3, 255, 0)

            intersection_regions = np.where(label == 3, 1, 0)

            if self.label_type == "Artery":
                region_pos, region_neg = artery, vein
            elif self.label_type == "Vein":
                region_pos, region_neg = vein, artery
            # global
            selected_coords_pos, selected_coords_neg = [], []

            select_labelmap = np.zeros_like(region_pos)

            labelmaps, connected_num = ndimage.label(region_pos, structure=structure)
            for ci in range(1, connected_num+1):
                connected_component = np.where(labelmaps == ci, 1, 0)
                coord_lst = list(zip(*np.where(connected_component == 1)))
                if len(coord_lst) > self.min_area:
                    selected_coords_pos += random.sample(coord_lst, 3)
                    select_labelmap += connected_component
            
            labelmaps, connected_num = ndimage.label(region_neg, structure=structure)
            for ci in range(1, connected_num+1):
                connected_component = np.where(labelmaps == ci, 1, 0)
                coord_lst = list(zip(*np.where(connected_component == 1)))
                if len(coord_lst) > self.min_area:
                    selected_coords_neg += random.sample(coord_lst, 3)
            
            selected_coords_pos = [(y, x) for x, y in selected_coords_pos]
            selected_coords_neg = [(y, x) for x, y in selected_coords_neg]

            prompt_points = selected_coords_pos + selected_coords_neg
            prompt_types = len(selected_coords_pos) * [1] + len(selected_coords_neg) * [0]
        
        return np.array(prompt_points), np.array(prompt_types), np.array([select_labelmap], dtype=float)

if __name__=="__main__":
    # "LargeVessel", "FAZ", "Capillary", "Artery", "Vein"
    label_type = "Capillary"
    pg = PromptGeneration(
            random_seed=42,
            neg_range=(0, 9), 
            positive_num=0, 
            negative_num=0,
            is_local=True,
            label_type=label_type,
        )
    for sample_id in tqdm(range(10001, 10301)):
        image_path = "datasets/OCTA-500/OCTA_6M/ProjectionMaps/OCTA(OPL_BM)/{}.bmp".format(sample_id)
        label_path = "datasets/OCTA-500/OCTA_6M/GT_{}/{}.bmp".format(label_type, sample_id)
        image, label = cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        pg.draw_generated_label(image, label, str(sample_id), "prompt_generation/{}.png".format(sample_id))
        break
    
    # for sample_id in tqdm(range(10301, 10501)):
    #     image_path = "datasets/OCTA-500/OCTA_3M/ProjectionMaps/OCTA(OPL_BM)/{}.bmp".format(sample_id)
    #     label_path = "datasets/OCTA-500/OCTA_3M/GT_{}/{}.bmp".format(label_type, sample_id)
    #     image, label = cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    #     pg.draw_generated_label(image, label, str(sample_id), "prompt_generation/{}.png".format(sample_id))