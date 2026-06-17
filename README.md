# A lightweight perceptron of interlaced spatial patterns for image classification

**Abstract**: Addressing grouped dilation features (GDFs) has improved the learning ability of lightweight models in image representation. However, it is lack of dilation patterns in diversity as well as the superposition of two tensors in the GDF-based perceptron would lead to a sharp increase of computational complexity due to the #channels on the double. To mitigate those issues, a lightweight model is proposed by addressing to two efficient concepts as follows. First, a perceptive block of stretched spatial features (SSF) is formed in consideration of diverse dilation features that are cumulated instead of the tensor superposition in the GDF-based block. A shallow backbone is then introduced by a sequential execution of the perceptive blocks to exploit the SSF-based patterns for the learning process (named SSFNet). Experimental results have verified the prominent efficacy of the proposed SSF-based network in comparison with GDF-based models as well as other existing lightweight ones. Specifically, the performance has boosted by up to from 5% to 7% on Stanford Dogs. Implementation code of SSFNet is available at https://github.com/CypriumCuprum/SSFNet.


## Installation

```bash
git clone 
cd 
conda env create -f environment.yml
conda activate ssfnet
```

## Data Preparation
About CIFAR, Places365, Stanford Dogs: automatically download with flag ```--download```\
About ImageNet-1K, ImageNet-100, ReIN, and RePLs: Manual download (place under  ```../../datasets/```):

## Training
```bash
python SSFNet_{dataset_name}.py
# dataset_name: see in the root dir
```

## Model Evaluation
To evaluate a trained model checkpoint instead of training from scratch, use the  -e  or  --evaluate  flag:
```bash  
python SSFNet_{dataset_name}.py -e
```