# torch
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn import DataParallel

# SAM
from segment_anything import *
from sam_lora_image_encoder import LoRA_Sam
from segment_anything.utils.transforms import ResizeLongestSide

# training
import numpy as np
from dataset import Octa500_Dataset
from options import *
from statistics import *
from loss_functions import *
from display import *
from metrics import MetricsStatistics

# others
import os, random, time, GPUtil
from tqdm import tqdm
from collections import *

parser = argparse.ArgumentParser(description='training arguments')
add_training_parser(parser)
add_octa500_2d_parser(parser)
args = parser.parse_args()

class TrainManager_OCTA:
    def __init__(self, 
                 dataset_train, 
                 dataset_val, 
                 parameters = ["3M", "Artery", 100, True, "vit_b", "Remark"], 
                 device_ids="0"):
        self.dataset_train, self.dataset_val = dataset_train, dataset_val
        
        self.fov, self.label_type, self.epochs, self.is_local, self.model_type, ppn, pnn, self.remark = parameters

        self.time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])
        

        self.record_dir = "results/{}/{}".format(self.time_str, "_".join(map(str, parameters)))
        print(self.record_dir)
        self.cpt_dir = "{}/checkpoints".format(self.record_dir)

        os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
        self.device_ids = device_ids
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to_cuda = lambda x: x.to(torch.float).to(self.device)

        if not os.path.exists(self.cpt_dir): os.makedirs(self.cpt_dir)

        if self.model_type == "vit_h":
            sam = sam_model_registry["vit_h"](checkpoint="sam_weights/sam_vit_h_4b8939.pth")
        elif self.model_type == "vit_l":
            sam = sam_model_registry["vit_l"](checkpoint="sam_weights/sam_vit_l_0b3195.pth")
        else:
            sam = sam_model_registry["vit_b"](checkpoint="sam_weights/sam_vit_b_01ec64.pth")

        self.sam_transform = ResizeLongestSide(224) if self.model_type == "vit_b" else ResizeLongestSide(1024)

        lora_sam = LoRA_Sam(sam, 4).cuda()
        self.model = DataParallel(lora_sam).to(self.device)
        torch.save(self.model.state_dict(), '{}/init.pth'.format(self.cpt_dir))

        self.loss_func = DiceLoss() 
        if self.label_type in ["Artery", "Vein", "LargeVessel"]: 
            self.loss_func = lambda x, y: 0.8 * DiceLoss()(x, y) + 0.2 * clDiceLoss()(x, y)
    
    def get_dataloader(self):
        indice_split = {
            "3M" : [(0, 140), (140, 150), (150, 200)],
            "6M" : [(0, 180), (180, 200), (200, 300)]
        }[self.fov]

        if self.is_local:
            sample_counter = self.dataset_val.sample_counter
            n1, n2, n3 = sample_counter["training"], sample_counter["validation"], sample_counter["test"]
            indice_split = {
                "3M" : [(0, 140), (n1, n1+n2), (n1+n2, n1+n2+n3)],
                "6M" : [(0, 180), (n1, n1+n2), (n1+n2, n1+n2+n3)]
            }[self.fov]

        batch_size = len(self.device_ids.split(","))

        train_sampler = range(len(self.dataset_train))[indice_split[0][0]:indice_split[0][1]]
        val_sampler = range(len(self.dataset_val))[indice_split[1][0]:indice_split[1][1]]
        test_sampler = range(len(self.dataset_val))[indice_split[2][0]:indice_split[2][1]]
        
        train_loader = DataLoader(self.dataset_train, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(self.dataset_val, batch_size=1, sampler=val_sampler)
        test_loader = DataLoader(self.dataset_val, batch_size=1, sampler=test_sampler)
        
        return train_loader, val_loader, test_loader

    def reset(self):
        self.model.load_state_dict(torch.load('{}/init.pth'.format(self.cpt_dir)))
        pg = [p for p in self.model.parameters() if p.requires_grad] # lora parameters
        self.optimizer = optim.AdamW(pg, lr=1, weight_decay=1e-4)
        epoch_p = self.epochs // 5
        lr_lambda = lambda x: max(1e-5, args.lr * x / epoch_p if x <= epoch_p else args.lr * 0.98 ** (x - epoch_p))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def record_performance(self, train_loader, val_loader, test_loader, epoch, metrics_statistics):
        save_dir = "{}/{:0>4}".format(self.record_dir, epoch)
        torch.save(self.model.state_dict(), '{}/{:0>4}.pth'.format(self.cpt_dir, epoch))

        metrics_statistics.metric_values["learning rate"].append(self.optimizer.param_groups[0]['lr'])

        def record_dataloader(dataloader, loader_type="val", is_complete=True):
            with torch.no_grad():
                for images, prompt_points, prompt_type, selected_components, sample_ids in dataloader:
                    images, labels, prompt_type = map(self.to_cuda, (images, selected_components, prompt_type))
                    images, original_size, prompt_points = self.make_prompts(images, prompt_points)
                    preds = self.model(images, original_size, prompt_points, prompt_type)
                    metrics_statistics.metric_values["loss_"+loader_type].append(self.loss_func(preds, labels).cpu().item())

                    if is_complete:
                        preds = torch.gt(preds, 0.8).int()
                        sample_id = str(sample_ids[0])

                        image, label, pred = map(lambda x:x[0][0].cpu().detach(), (images, labels, preds))
                        prompt_points, prompt_type = prompt_points[0].cpu().detach(), prompt_type[0].cpu().detach()
                        prompt_info = np.concatenate((prompt_points, prompt_type[:,np.newaxis]), axis=1).astype(int)
                        metrics_statistics.cal_epoch_metric(
                            args.metrics, "{}-{}".format(self.label_type, loader_type), label.int(), pred.int())
                        
                        if not os.path.exists(save_dir): os.makedirs(save_dir)
                        save_sample_func = lambda x, y: np.save("/".join([save_dir,\
                                            "{}_{}_{}.npy".format(self.label_type, x, sample_id)]), y)
                        save_items = {"sample":image / 255, "label":label, "prompt_info":prompt_info, "pred":pred}
                        for x, y in save_items.items(): save_sample_func(x, y)

        record_dataloader(train_loader, "train", False)
        record_dataloader(val_loader, "val", True)
        record_dataloader(test_loader, "test", True)

        metrics_statistics.record_result(epoch)
    
    def train(self):
        train_loader, val_loader, test_loader = self.get_dataloader()
        self.reset()
        metrics_statistics = MetricsStatistics(save_dir=self.record_dir)
        self.record_performance(train_loader, val_loader, test_loader, 0, metrics_statistics)
        show_pos = int(self.device_ids[-1])
        for epoch in tqdm(range(1, self.epochs+1), desc="training-{}".format(show_pos), position=show_pos, leave=True):
            for images, prompt_points, prompt_type, selected_components, sample_ids in train_loader:
                images, labels, prompt_type = map(self.to_cuda, (images, selected_components, prompt_type))
                images, original_size, prompt_points = self.make_prompts(images, prompt_points)
                self.optimizer.zero_grad()
                preds = self.model(images, original_size, prompt_points, prompt_type)
                self.loss_func(preds, labels).backward()
                self.optimizer.step()
            self.scheduler.step()
            if epoch % args.check_interval == 0: 
                self.record_performance(train_loader, val_loader, test_loader, epoch, metrics_statistics)
        metrics_statistics.close()
        
    def make_prompts(self, images, prompt_points):
        original_size = tuple(images.shape[-2:])
        images = self.sam_transform.apply_image_torch(images)
        prompt_points = self.sam_transform.apply_coords_torch(prompt_points, original_size)

        return images, original_size, prompt_points

## training strategies
def train_single(dataset_params, training_params, device):
    gpus = GPUtil.getGPUs()
    for gpu_id, gpu in enumerate(gpus):
        if str(gpu_id) in device.split(","):
            print(f"GPU {gpu_id}:")
            print(f"  Total: {gpu.memoryTotal // 1024} GB")
            print(f"  Allocated: {gpu.memoryUsed // 1024} GB")
            print(f"  Reserved: {gpu.memoryFree // 1024} GB")
    random_seed = 42

    dataset_train = Octa500_Dataset(*dataset_params, True, random_seed)
    dataset_val = Octa500_Dataset(*dataset_params, False, random_seed)

    train_manager = TrainManager_OCTA(dataset_train, dataset_val, training_params, device)
    train_manager.train()

def train_profile():
    ppn, pnn = args.prompt_positive_num, args.prompt_negative_num
    dataset_params = [args.fov, args.label_type, ppn, pnn, args.is_local]
    training_params = [args.fov, args.label_type, args.epochs, args.is_local, args.model_type, ppn, pnn, args.remark]
    train_single(dataset_params, training_params, args.device)

if __name__=="__main__":
    train_profile()
