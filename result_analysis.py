import os
import pandas as pd
import numpy as np
import shutil

from tqdm import tqdm
from collections import defaultdict

from display import display_plot

class ResultAnalysis:
    def __init__(self, 
                 result_dir="results",
                 start_time_str="2024-01",
                 end_time_str="2024-04"):
        self.result_dir = result_dir
        self.start_time_str = start_time_str
        self.end_time_str = end_time_str
        self.valid_dirs = []
        self.get_valid_results()

    def get_valid_results(self, filter=""):
        valid_dirs = []
        for datetime_dir in sorted(os.listdir(self.result_dir)):
            if self.start_time_str <= datetime_dir <= self.end_time_str:
                for task_dir in sorted(os.listdir("{}/{}".format(self.result_dir, datetime_dir))):
                    if not filter or filter in task_dir:
                        valid_dirs.append("{}/{}/{}".format(self.result_dir, datetime_dir, task_dir))
        self.valid_dirs = valid_dirs
        return valid_dirs
    
    def check_failure_results(self):
        failure_results = []
        for valid_dir in self.valid_dirs:
            ms_file_path = "{}/metrics_statistics.xlsx".format(valid_dir)
            if os.path.exists(ms_file_path):
                df = pd.read_excel(ms_file_path)
                task_options = valid_dir.split("/")[-1]
                total_epochs = int(task_options.split("_")[2])
                trained_epochs = int(list(df["epoch"])[-1])
                if total_epochs != trained_epochs:
                    task_dir = valid_dir.split("/")[1]
                    failure_results.append([task_dir, total_epochs, trained_epochs])
            else:
                task_dir = valid_dir.split("/")[1]
                failure_results.append([task_dir])
        return failure_results

    def delete_failure_results(self):
        for failure_result in tqdm(self.check_failure_results(), desc="remove invalid"):
            shutil.rmtree("{}/{}".format(self.result_dir, failure_result[0]))
        self.get_valid_results()
    
    def get_test_result(self, timestamp=""):
        task_dir = "{}/{}".format(self.result_dir, timestamp)
        rnt = []
        if os.path.exists(task_dir):
            task_options = os.listdir(task_dir)[0]
            # print("task options :", task_options)
            df = pd.read_excel("{}/{}/metrics_statistics.xlsx".format(task_dir, task_options))
            val_max_index = df["loss_val"].idxmin()
            for k in df:
                if "test" in k: 
                    # print("{} : {}".format(k, df[k][val_max_index]))
                    if "-" in k:
                        metric, value = k.split()[0], round(df[k][val_max_index], 4)
                        rnt.append((metric, value))
        return rnt

'''

'''
if __name__=="__main__":
    ra = ResultAnalysis()
    # ra.delete_failure_results()
    g = defaultdict(list)

    # Artery, Vein, LargeVessel, FAZ, Capillary
    label_type = "Vein"

    # "Bifurcation", "Endpoint", "Intersection", "All"
    # "Special"
    
    tags = set(["All"])
    bans = set(["Bifurcation", "Endpoint", "Intersection"]) - tags

    for x in ra.get_valid_results():
        # if "True" in x and label_type in x and "_3_3" in x:
        # if "6M" in x and "SpecialPoints" in x and label_type in x and (set(x.split("_")) & tags == tags) and (set(x.split("_")) & bans == set()):
        # if "6M" in x and label_type in x and "vit_h" in x and "Retro" in x:
        if "3M" in x and label_type in x and "vit_h" in x and "True" in x and "_0_6" in x:
            print(x)
            datetime_dir = x.split("/")[1]
            g["condition"].append(x.split("/")[-1])
            for k, v in ra.get_test_result(datetime_dir):
                g[k].append(v)

    df = pd.DataFrame(g)
    df_sorted = df.sort_values(by='condition')

    print(df_sorted)
    
    # # print("Dice", list(df_sorted["Dice-{}-test".format(label_type)]))
    # # print("Jaccard", list(df_sorted["Jaccard-{}-test".format(label_type)]))
    # # print("Hausdorff", list(df_sorted["Hausdorff-{}-test".format(label_type)]))
 
    # g_label = defaultdict(list)
    # # g_label["Dice-3M"] = [0.775, 0.8226, 0.8413, 0.8511, 0.8587, 0.8636, 0.8638, 0.8685]
    # # g_label["Dice-6M"] = [0.7073, 0.7373, 0.7787, 0.7955, 0.8022, 0.8129, 0.8155, 0.8192]

    # # g_label["Jaccard-3M"] = [0.6721, 0.7162, 0.7424, 0.7528, 0.7628, 0.7696, 0.7692, 0.7753]
    # # g_label["Jaccard-6M"] = [0.5788, 0.6097, 0.6541, 0.6738, 0.6823, 0.695, 0.698, 0.7022]

    # # g_label["Hausdorff-3M"] = [2.3963, 2.2691, 2.2104, 2.1885, 2.1862, 2.1536, 2.1668, 2.172]
    # # g_label["Hausdorff-6M"] = [4.6827, 3.5971, 3.3942, 3.304, 3.2674, 3.189, 3.2011, 3.2009]

    # # g_label["Dice-3M"] = [0.9723, 0.9795, 0.9813, 0.9809, 0.9824, 0.9824]
    # # g_label["Dice-6M"] = [0.8852, 0.8999, 0.9126, 0.9116, 0.9109, 0.9109]

    # # g_label["Jaccard-3M"] = [0.9556, 0.9604, 0.9636, 0.963, 0.9659, 0.9659]
    # # g_label["Jaccard-6M"] = [0.826, 0.8376, 0.8567, 0.8557, 0.854, 0.854]

    # g_label["Hausdorff-3M"] = [2.6962, 2.6994, 2.5795, 2.6075, 2.5791, 2.5791]
    # g_label["Hausdorff-6M"] = [3.4175, 3.1782, 3.0198, 3.0675, 3.0811, 3.0811]

    
    # print(g_label)
    # display_plot(
    #     conditions=["{} / {}".format(x, x) for x in range(0, 6)],
    #     g_label = g_label,
    #     label_type=label_type,
    #     xlabel="Number of prompt points: positive / negative",
    #     ylabel="",
    #     ylim=(0.1, 0.2),
    #     title="Local mode-{}".format(label_type),
    # )

'''
Global
- Dice
    g_label["RV-3M"] = [0.914, 0.9138, 0.9141, 0.9128, 0.9157, 0.9151]
    g_label["RV-6M"] = [0.8881, 0.8886, 0.8887, 0.8886, 0.8888, 0.8892]
    g_label["FAZ-3M"] = [0.9723, 0.9795, 0.9813, 0.9809, 0.9824, 0.9824]
    g_label["FAZ-6M"] = [0.9152, 0.9299, 0.9426, 0.9416, 0.9409, 0.9409]
    g_label["Capillary-3M"] = [0.8719, 0.8641, 0.871, 0.8757, 0.8711, 0.8755]
    g_label["Capillary-6M"] = [0.806, 0.809, 0.8089, 0.8086, 0.8091, 0.8082]
    g_label["Artery-3M"] = [0.8863, 0.8853, 0.8893, 0.885, 0.8895, 0.8869]
    g_label["Artery-6M"] = [0.8557, 0.8569, 0.8577, 0.8563, 0.858, 0.858]
    g_label["Vein-3M"] = [0.8858, 0.8872, 0.8855, 0.8869, 0.8821, 0.8865]
    g_label["Vein-6M"] = [0.8572, 0.8567, 0.8593, 0.8576, 0.8575, 0.8599]

- Jaccard
    g_label["RV-3M"] = [0.8423, 0.842, 0.8425, 0.8403, 0.8452, 0.8442]
    g_label["RV-6M"] = [0.7997, 0.8006, 0.8008, 0.8005, 0.8008, 0.8015]
    g_label["FAZ-3M"] = [0.9556, 0.9604, 0.9636, 0.963, 0.9659, 0.9659]
    g_label["FAZ-6M"] = [0.856, 0.8776, 0.8967, 0.8957, 0.894, 0.894]
    g_label["Capillary-3M"] = [0.7734, 0.7612, 0.772, 0.7793, 0.7722, 0.7791]
    g_label["Capillary-6M"] = [0.6763, 0.6807, 0.6804, 0.6801, 0.6807, 0.6794]
    g_label["Artery-3M"] = [0.7969, 0.7955, 0.8017, 0.7949, 0.802, 0.7978]
    g_label["Artery-6M"] = [0.7496, 0.7513, 0.7521, 0.7504, 0.753, 0.7528]
    g_label["Vein-3M"] = [0.796, 0.7982, 0.7954, 0.7975, 0.7899, 0.7969]
    g_label["Vein-6M"] = [0.7516, 0.7512, 0.7548, 0.7524, 0.7522, 0.7558]

- Hausdorff
    g_label["RV-3M"] = [4.2373, 4.2199, 4.2231, 4.209, 4.2075, 4.1912]
    g_label["RV-6M"] = [5.4874, 5.4657, 5.4436, 5.424, 5.4388, 5.4137]
    g_label["FAZ-3M"] = [2.6962, 2.6994, 2.5795, 2.6075, 2.5791, 2.5791]
    g_label["FAZ-6M"] = [3.2175, 2.9782, 2.8198, 2.8675, 2.8811, 2.8811]
    g_label["Capillary-3M"] = [7.6139, 7.8154, 7.635, 7.5977, 7.6473, 7.6283]
    g_label["Capillary-6M"] = [9.5096, 9.5887, 9.5338, 9.5569, 9.5557, 9.5002]
    g_label["Artery-3M"] = [4.1499, 4.1711, 4.0511, 4.087, 4.086, 4.1248]
    g_label["Artery-6M"] = [5.0317, 5.0759, 5.0779, 5.0822, 5.064, 5.0775]
    g_label["Vein-3M"] = [3.5728, 3.5825, 3.6299, 3.6417, 3.6396, 3.612]
    g_label["Vein-6M"] = [4.9132, 5.0479, 4.9702, 4.9726, 5.085, 4.9363]
'''

'''
Local
- Dice
    g_label["RV-3M"] = [0.8131, 0.8414, 0.8542, 0.8643, 0.8648, 0.8701, 0.8713, 0.8701]
    g_label["RV-6M"] = [0.6349, 0.7499, 0.7703, 0.7773, 0.7872, 0.7956, 0.7927, 0.7978]
    g_label["Artery-3M"] = [0.7759, 0.8189, 0.8391, 0.8421, 0.8571, 0.858, 0.8578, 0.8667]
    g_label["Artery-6M"] = [0.6932, 0.7361, 0.7691, 0.7826, 0.7883, 0.7996, 0.7968, 0.7881]
    g_label["Vein-3M"] = [0.775, 0.8226, 0.8413, 0.8511, 0.8587, 0.8636, 0.8638, 0.8685]
    g_label["Vein-6M"] = [0.7073, 0.7373, 0.7787, 0.7955, 0.8022, 0.8129, 0.8155, 0.8192]
- Jaccard
    g_label["RV-3M"] = [0.7149, 0.746, 0.7601, 0.7732, 0.7748, 0.7805, 0.7815, 0.7814]
    g_label["RV-6M"] = [0.519, 0.6274, 0.6474, 0.6575, 0.6663, 0.676, 0.6736, 0.6788]
    g_label["Artery-3M"] = [0.6621, 0.7122, 0.7359, 0.7393, 0.7587, 0.7603, 0.7592, 0.7715]
    g_label["Artery-6M"] = [0.5687, 0.6139, 0.6462, 0.6603, 0.6666, 0.6804, 0.6757, 0.6686]
    g_label["Vein-3M"] = [0.6721, 0.7162, 0.7424, 0.7528, 0.7628, 0.7696, 0.7692, 0.7753]
    g_label["Vein-6M"] = [0.5788, 0.6097, 0.6541, 0.6738, 0.6823, 0.695, 0.698, 0.7022]
- Hausdorff
    g_label["RV-3M"] = [2.3963, 2.2691, 2.2104, 2.1885, 2.1862, 2.1536, 2.1668, 2.172]
    g_label["RV-6M"] = [4.6827, 3.5971, 3.3942, 3.304, 3.2674, 3.189, 3.2011, 3.2009]
    g_label["Artery-3M"] = [2.6528, 2.4938, 2.4687, 2.4491, 2.3566, 2.4182, 2.3661, 2.3473]
    g_label["Artery-6M"] = [4.2646, 3.9493, 3.5537, 3.4994, 3.4401, 3.3929, 3.4313, 3.506]
    g_label["Vein-3M"] = [2.2893, 2.1027, 2.0417, 1.9895, 1.9984, 1.9788, 1.9539, 1.9436]
    g_label["Vein-6M"] = [3.329, 3.1199, 2.9395, 2.8679, 2.7624, 2.7561, 2.7113, 2.7016]
'''