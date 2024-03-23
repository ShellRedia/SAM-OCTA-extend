# SAM-OCTA_extend (中文版)

这是之前SAM-OCTA项目补充了一些实验的拓展版本，SAM-OCTA的链接：
https://github.com/ShellRedia/SAM-OCTA

~~准备投篇期刊，之前有审稿人建议弄两个仓库，避免争议，所弄了个新的~~

## 1. 预训练权重配置

这是一个使用 LoRA 对 SAM 进行微调，并在 OCTA 图像上执行分割任务的项目, 使用 **PyTorch** 构建。

首先，您应该将一个预训练的权重文件放入 **sam_weights** 文件夹中。预训练权重的下载链接如下:

vit_h (default): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 

vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

并修改 **options.py** 中的对应配置项。

    ...
    parser.add_argument("-model_type", type=str, default="vit_h")
    ...

## 2. 数据集存放形式

~~这次我学聪明了，直接把这个复杂路径下的文件都留了前5个，简直一目了然啊~~

见 **datasets** 文件夹中数据的存放形式

如果需要完整的数据集，需要联系 **OCTA_500** 数据集的作者。

**OCTA-500**'s related paper: https://arxiv.org/abs/2012.07261

示例结果和分割指标将被记录在 **results** 文件夹中（如果不存在，则这个文件夹将被自动创建）。

如果您需要对预测结果进行可视化，请使用 **display.py** 文件。由于结果文件夹是按时间生成的，需要对这一行代码进行替换。生成的图像存放在 **sample_display** 文件夹中。

    ...
        result_dir = r"results\2024-03-18-17-05-26\3M_Vein_50_True_vit_b_Intersection_Cross\0010" # Your result dir
    ...

这里我额外做了一点功能，可以通过解注释下面的语句来对比两次实验的结果。

## 3 相关配置

基本和 [SAM-OCTA](https://github.com/ShellRedia/SAM-OCTA) 相一致，不过为了实验的完整性，多了一个 **-point_type** 用于指定特殊点的类型，这一点会在下一节提供更为详细的说明。

## 4 训练模式

结合论文，本项目提供了三种训练模式：

**随机选择**：

    python train_special_points.py

从分割目标上随机选取若干正提示点，并在周围随机选取若干负提示点。

**特殊点标记**

    python train_special_points.py

选取血管上的分叉点、端点以及动脉、静脉的交叉点作为正提示点，这些提示点由稀疏标注得出，有需要的话我会考虑传一份到网盘。

**交叉抑制**

    python train_cross.py

选取了非目标血管区域作为负提示点，从而验证负提示点对于非目标区域的一直作用。从结果而言作用不大，目测是提示点都是范围作用，需要做更为精确的约束。

## 5 相关预印本

https://arxiv.org/abs/2310.07183