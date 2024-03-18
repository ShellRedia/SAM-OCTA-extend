import argparse

parser = argparse.ArgumentParser(description='training argument values')

def add_training_parser(parser):
    parser.add_argument("-device", type=str, default="0")
    parser.add_argument("-epochs", type=int, default=50)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-check_interval", type=int, default=5)
    parser.add_argument("-prompt_positive_num", type=int, default=3)
    parser.add_argument("-prompt_negative_num", type=int, default=3)
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("-is_local", type=bool, default=True)
    parser.add_argument("-point_type", type=str, default="Intersection")
    parser.add_argument("-remark", type=str, default="Cross")

def add_octa500_2d_parser(parser):
    parser.add_argument("-fov", type=str, default="3M")
    parser.add_argument("-label_type", type=str, default="Vein") # "LargeVessel", "FAZ", "Capillary", "Artery", "Vein"
    parser.add_argument("-metrics", type=list, default=["Dice", "Jaccard", "Hausdorff"])