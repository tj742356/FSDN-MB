# <p align=center>`Medical Image Segmentation`</p>


:fire::fire: This is an official repository of our work on medical image segmentation.:fire::fire:

> If you have any questions about our work, feel free to contact me via e-mail (zhangdeping@stu.gpnu.edu.cn).

## Get Start
> Our experiments are based on ubuntu, and windows is not recommended.
> 
**0. Install**

```
conda create --name FSDN-MB python=3.8 -y
conda activate FSDN-MB
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

cd mmsegmentation
pip install -v -e .
pip install ftfy
pip install regex
pip install einops
```

The following methods can be used to verify that the experimental environment is successfully set up:
```
1. mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
2. python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg
```
After the preceding two steps are successfully run, if the result.png file is generated under the mmsegmentation folder, the environment is successfully created.The result.png as shown in the following.

<p align="center"><img width="800" alt="image" src="https://github.com/haoshao-nku/medical_seg/blob/master/mmsegmentation/demo/result.jpg"></p> 

**1. Dataset**
> The dataset used in the experiment can be obtained in the following methods:
- For polyp segmentation task: [Polypseg](https://github.com/DengPingFan/PraNet): including Kvasir, - CVC-ClinicDB, CVC-ColonDB, EndoScene and ETIS dataset.


**2. Experiments**
We recommend that you place the project folder in a location such as a solid state drive, and put the checkpoint files generated from the experiment on a mechanical hard drive to save space, so you can choose to create a soft connection. Specific practices are as follows:

> ln -s   "mechanical hard disk path"  /medical_seg/mmsegmentation/work_dirs

If your hardware resources are relatively rich, ignore this advice.

> **Note: Our experiment is implemented based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). The environment configuration can also refer to the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), and questions about the entire project can refer to the [official documentation](https://mmsegmentation.readthedocs.io/zh-cn/latest/).**

## Our Work

###多分支架构的频域-空间域双域分割网络
###FSDN-MB: Frequency-Spatial Dual-Domain Network with Multi-Branch Architecture

#### **Abstract**
结直肠癌（CRC）是全球范围内导致死亡的主要癌症之一，其早期发现和治疗对于降低死亡率至关重要。结肠镜检查是检测癌前息肉的关键方法，但自动化的息肉检测在准确性和效率上存在挑战。目前，深度学习技术在医学图像分析中展现出巨大潜力，尤其是在息肉的自动化检测和分割方面。
然而，现有模型在处理全局与局部特征时存在不连续性，且多忽视频域信息，影响了分割的精度和鲁棒性。针对这些挑战，我们开发了一种创新的频域-空间域双域分割网络（FSDN-MB），该网络采用多分支架构，集成了局部-全局空间处理模块（LGSPM）和自适应特征选择模块（SAEM），以及多级多尺度跨域融合模块（MMCFM）。
FSDN-MB的设计旨在通过精确的图像分割辅助医生进行息肉的检测和诊断，提高结直肠癌筛查的准确性和效率。
在Kvasir-SEG、CVC-ClinicDB和EndoScene数据集上的测试结果表明，FSDN-MB在Dice系数、平均交并比（mIoU）、准确率（ACC）、召回率（Recall）和精确度（Precision）等关键指标上均表现优异。
这一成果不仅提升了息肉分割的性能，也为临床医生提供了一个强有力的工具，以支持更快速、更准确的诊断决策，具有显著的临床应用价值和社会效益。
#### Architecture

<p align="center">
    <img src="https://github.com/haoshao-nku/medical_seg/blob/master/fig/pipline_polyper.png"/> <br />
    <em> 
    Figure 1: Overall architecture of Polyper. We use the Swin-T from Swin Transformer as the encoder. The decoder is divided into two main stages. The first potential boundary extraction (PBE) stage aims to capture multi-scale features from the encoder, which are then aggregated to generate the initial segmentation results. Next, we extract the predicted polyps' potential boundary and interior regions using morphology operators. In the second boundary sensitive refinement (BSR) stage, we model the relationships between the potential boundary and interior regions to generate better segmentation results.
    </em>
</p>


<p align="center">
    <img src="https://github.com/haoshao-nku/medical_seg/blob/master/fig/refine_polyper.png"/> <br />
    <em> 
    Figure 2: Detailed structure of boundary sensitive attention (BSA) module. This process is separated into two parallel branches, which systematically capitalize on the distinctive attributes of polyps at various growth stages, both in terms of spatial and channel characteristics. `B' and `M' indicate the number of pixels in the boundary and interior polyp regions within an input of size H*W and C channels.
    </em>
</p>

#### Experiments

> For training, testing and other details can be found at **/medical_seg/mmsegmentation/local_config/Polyper-AAAI2024/readme.md**.

### [MCANet: Medical Image Segmentation with Multi-Scale Cross-Axis Attention](https://arxiv.org/abs/2312.08866)

> **Authors:**
> [Hao Shao](https://scholar.google.com/citations?hl=en&user=vB4DPYgAAAAJ), [Quansheng Zeng](), [Qibin Hou](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=en&oi=ao), &[Jufeng Yang](https://scholar.google.com/citations?user=c5vDJv0AAAAJ&hl=en&oi=ao).

#### **Abstract**


Efficiently capturing multi-scale information and building long-range dependencies among pixels are essential for medical image segmentation because of the various sizes and shapes of the lesion regions or organs. In this paper, we present Multi-scale Cross-axis Attention (MCA) to solve the above challenging issues based on the efficient axial attention. Instead of simply connecting axial attention along the horizontal and vertical directions sequentially, we propose to calculate dual cross attentions between two parallel axial attentions to capture global information better. To process the significant variations of lesion regions or organs in individual sizes and shapes, we also use multiple convolutions of strip-shape kernels with different kernel sizes in each axial attention path to improve the efficiency of the proposed MCA in encoding spatial information. We build the proposed MCA upon the MSCAN backbone, yielding our network, termed MCANet. Our MCANet with only 4M+ parameters performs even better than most previous works with heavy backbones (e.g., Swin Transformer) on four challenging tasks, including skin lesion segmentation, nuclei segmentation, abdominal multi-organ segmentation, and polyp segmentation.

#### Architecture



<p align="center">
    <img src="https://github.com/haoshao-nku/medical_seg/blob/master/fig/pipeline-MCANet.png"/> <br />
    <em> 
    Figure 1: Overall architecture of the proposed MCANet. We take the MSCAN network proposed in SegNeXt as our encoder because of its capability of capturing multi-scale features. The feature maps from the last three stages of the encoder are combined via upsampling and then concatenated as the input of the decoder. Our decoder is based on multi-scale cross-axis attention, which takes advantage of both multi-scale convolutional features and the axial attention.
    </em>
</p>



<p align="center">
    <img src="https://github.com/haoshao-nku/medical_seg/blob/master/fig/decoder-MCANet.png"/> <br />
    <em> 
    Figure 2: Detailed structure of the proposed multi-scale cross-axis attention decoder. Our decoder contains two parallel paths, each of which contains multi-scale 1D convolutions and cross-axis attention to aggregate the spatial information. Note that we do not add any activation functions in decoder.
    </em>
</p>


#### Experiments

> For training, testing and other details can be found at **/medical_seg/mmsegmentation/local_config/MCANet/readme.md**.


## Acknowlegement

Thanks [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) providing a friendly codebase for segmentation tasks. And our code is built based on it.

## Reference
You may want to cite:
```
@inproceedings{shao2024polyper,
  title={Polyper: Boundary Sensitive Polyp Segmentation},
  author={Shao, Hao and Zhang, Yang and Hou, Qibin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4731--4739},
  year={2024}
}

@article{shao2023mcanet,
  title={MCANet: Medical Image Segmentation with Multi-Scale Cross-Axis Attention},
  author={Shao, Hao and Zeng, Quansheng and Hou, Qibin and Yang, Jufeng},
  journal={arXiv preprint arXiv:2312.08866},
  year={2023}
}
```




### License

Code in this repo is for non-commercial use only.
