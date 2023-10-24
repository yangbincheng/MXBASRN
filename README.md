# Single Image Super-Resolution with Sequential Multi-axis Blocked Attention

This repository is for MXBASRN introduced in the following paper

Yang, B., Wu, G. (2023). Single Image Super-Resolution with Sequential Multi-axis Blocked Attention. In: Iliadis, L., Papaleonidas, A., Angelov, P., Jayne, C. (eds) Artificial Neural Networks and Machine Learning – ICANN 2023. ICANN 2023. Lecture Notes in Computer Science, vol 14256. Springer, Cham. https://doi.org/10.1007/978-3-031-44213-1_12

The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [RCAN](https://github.com/yulunzhang/RCAN), tested on Ubuntu 16.04 environment (Python3.7, PyTorch_1.1.0, CUDA9.0) with Titan X/Xp/V100 GPUs.

## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction

Single image super-resolution is an ill-posed inverse problem which has no unique solution because the low resolution image can be mapped to many different undegraded high resolution images. Previous methods based on deep neural networks try to utilize non-local attention mechanisms to leverage self-similarity prior in natural images in order to tackle the ill-posedness of SISR and improve the performance for SISR. However, because non-local attention has a quadratic order computation complexity with respect to the number of attention locations and the very big spatial sizes of feature maps of SISR networks, the nonlocal attention mechanisms utilized in current methods can not achieve a good trade-off between global modelling capability of self-similarity to improve performance and lower computation complexity to be efficient and scalable. In this paper, we propose to utilize a sequential multiaxis blocked attention (S-MXBA) mechanism in a deep neural network (MXBASRN) to achieve a good trade-off between performance and efficiency for SISR. S-MXBA splits the input feature map into blocks of appropriate size to balance the size of each block and the number of all the blocks, then does non-local attention inside each block followed by non-local attention to the same relative locations across all blocks. In this way, MXBASRN both improves global modelling capability of selfsimilarity to boost performance and decreases computation complexity to sub-quadratic order to be more efficient and scalable. Experiments demonstrate MXBASRN works effectively and efficiently for SISR compared to state-of-the-art methods. Especially, MXBASRN achieves comparable performance to recent non-local attention based SISR methods of NLSN and ENLCN with about one-third parameters of them.

## Train
### Prepare training data

1. Download DIV2K training data (800 training images for x2, x3, x4) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/). 

2. Untar the download files.

3. Specify '--dir_data' in src/option.py to your directory to place DIV2K dataset.

For more informaiton, please refer to [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Begin to train

1. Cd to 'src', run the following scripts to train models.

    **You can use scripts in file 'demo.sh' to train models for our paper.**

    To train a fresh model using DIV2K dataset for x2 scale factor
    ```bash
    CUDA_VISIBLE_DEVICES=0 python main.py --model MXBASRN --save MXBASRN_BIX2 --scale 2 --n_resgroups 6 --n_resblocks 8 --n_feats 128 --ext sep --print_model --patch_size 128 --batch_size 16 --no_attention --n_threads 8 --shift_mean True --use_multi_axis_atten --multi_axis_atten_method pure --atten_noself --no_perfect_balanced_blocking --res_scale 1 --small_upsampling_head --small_upsampling_head_with_width_threshold --small_upsampling_head_with_width_threshold_value 64 --n_GPUs 1 --chop --dir_data /data/SR --with_layer_norm_channel --configurable_attention --configurable_attention_num 6 --reset
    ```

    To continue a unfinished model using DIV2K dataset for x2 scale factor
    ```bash
    CUDA_VISIBLE_DEVICES=0 python main.py --model MXBASRN --save MXBASRN_BIX2 --scale 2 --n_resgroups 6 --n_resblocks 8 --n_feats 128 --ext sep --print_model --patch_size 128 --batch_size 16 --no_attention --n_threads 8 --shift_mean True --use_multi_axis_atten --multi_axis_atten_method pure --atten_noself --no_perfect_balanced_blocking --res_scale 1 --small_upsampling_head --small_upsampling_head_with_width_threshold --small_upsampling_head_with_width_threshold_value 64 --n_GPUs 1 --chop --dir_data /data/SR --with_layer_norm_channel --configurable_attention --configurable_attention_num 6 --load MXBASRN_BIX2 --resume -1
    ```

## Test
### Quick start

1. Download benchmark dataset from [BaiduYun](https://pan.baidu.com/s/1Bl8TUHywC1HUHoamUFdCew) (code：20v5), place them in directory specified by '--dir_data' in src/option.py, untar it.

2. Download model for our paper from [BaiduYun](https://pan.baidu.com/s/1D1i8JHRBq64cxXal7_JQUQ?pwd=mwww) (code：mwww) and place them in 'experiment/'.

3. Cd to 'src', run the following scripts to test downloaded model.

    **You can use scripts in file 'demo.sh' to produce results for our paper.**

    To test a trained model for x2 scale factor
    ```bash
    CUDA_VISIBLE_DEVICES=0 python main.py --model MXBASRN --save MXBASRN_BIX2_test --scale 2 --n_resgroups 6 --n_resblocks 8 --n_feats 128 --ext sep --print_model --patch_size 128 --batch_size 16 --no_attention --n_threads 8 --shift_mean True --use_multi_axis_atten --multi_axis_atten_method pure --atten_noself --no_perfect_balanced_blocking --res_scale 1 --small_upsampling_head --small_upsampling_head_with_width_threshold --small_upsampling_head_with_width_threshold_value 64 --n_GPUs 1 --chop --dir_data /data/SR --with_layer_norm_channel --configurable_attention --configurable_attention_num 6 --pre_train ../experiment/MXBASRN_BIX2.pt --test_only --data_test Set5+Set14+B100+Urban100+Manga109 --save_results 
    ```

    To test a trained model for x2 scale factor using self ensemble
    ```bash
    CUDA_VISIBLE_DEVICES=0 python main.py --model MXBASRN --save MXBASRN_BIX2_test+ --scale 2 --n_resgroups 6 --n_resblocks 8 --n_feats 128 --ext sep --print_model --patch_size 128 --batch_size 16 --no_attention --n_threads 8 --shift_mean True --use_multi_axis_atten --multi_axis_atten_method pure --atten_noself --no_perfect_balanced_blocking --res_scale 1 --small_upsampling_head --small_upsampling_head_with_width_threshold --small_upsampling_head_with_width_threshold_value 64 --n_GPUs 1 --chop --dir_data /data/SR --with_layer_norm_channel --configurable_attention --configurable_attention_num 6 --pre_train ../experiment/MXBASRN_BIX2.pt --test_only --data_test Set5+Set14+B100+Urban100+Manga109 --save_results --self_ensemble
    ```

## Results
All the test results can be download from [BaiduYun](https://pan.baidu.com/s/1D1i8JHRBq64cxXal7_JQUQ?pwd=mwww) (code：mwww).

## Citation
If you find the code helpful in your resarch or work, please cite the following paper.
```
@InProceedings{10.1007/978-3-031-44213-1_12,
author="Yang, Bincheng
and Wu, Gangshan",
editor="Iliadis, Lazaros
and Papaleonidas, Antonios
and Angelov, Plamen
and Jayne, Chrisina",
title="Single Image Super-Resolution with Sequential Multi-axis Blocked Attention",
booktitle="Artificial Neural Networks and Machine Learning -- ICANN 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="136--148",
isbn="978-3-031-44213-1"
}
```

## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [RCAN](https://github.com/yulunzhang/RCAN). We thank the authors for sharing their code.
