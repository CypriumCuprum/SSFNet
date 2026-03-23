# A lightweight perceptron of interlaced spatial patterns for image classification

**Abstract**: Addressing dilated features has improved the learning ability of lightweight CNNbased networks in image representation. However, the performance is still at a modest level due to lack of global scattered features surrounding a local convolution region for their learning process. Addressing a convolutional multiple of large kernels for these global features seem not to be suitable for a lightweight model. To deal with these issues, we proposed a novel lightweight operator for interlacing patterns grouped dilation features (GDFs) has improved the learning ability of MobileNetV1 in image representation. Its enhanced version the completed block of GDF-based feature(CGDF), combines a new lightweight backbone with the popular squeeze-and-excitation (SE) operator. We recognize the potential of developing an active mechanism for internally combining features within a block, rather than relying on conventional propagation. Driven by this motivation, we propose a new lightweight model named modified dilation features (MDF-Net) with the following main contributions: i) A proposed fusion mechanism that actively combines dilation features, leading to a more effective receptive field while reducing learnable parameters compared to the GDF-block; ii) Applying a full residual mechanism instead of the partial residual used in the previous CGDF; iii) An adaptive backbone for MDF-Net that is deeper yet lighter than the CGDF backbone. Experimental results of MDF-Net on popular benchmark datasets for image classification tasks, along with supporting analyses, have demonstrated the effectiveness of MDF-Net compared to other lightweight models. Specifically, on the Stanford Dogs dataset, MDF-Net achieves 64 75% with 2.33M parameters, representing a 4% improvement in accuracy and 33% fewer parameters compared to CGDF-Net (60.86%, 3.53M parameters). Notably, the MDF-Net version with a 0.5 width multiplier achieves the smallest model size while maintaining comparable accuracy with other lightweight models.

## Installation

```bash
git clone 
cd 
conda env create -f environment.yml
conda activate mdfnet
```

## Data Preparation
About CIFAR, Places365, ImageNet: automatically download through torchvision
About Stanford Dogs: auto download (see file /models/datasets.py)

## Training
```bash
python MDF_{dataset_name}.py
# dataset_name: see in the root dir
```