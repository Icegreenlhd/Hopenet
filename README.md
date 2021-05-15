# Hopenet #

<div align="center">
  <img src="https://i.imgur.com/K7jhHOg.png" width="380"><br><br>
</div>
**Hopenet**是一个准确且易于使用的头部姿势估计网络。 在300W-LP数据集上对模型进行了训练，并在具有良好定性性能的真实数据上进行了测试。

有关方法和定量结果的详细信息，请检查CVPR会议论文[paper](https://arxiv.org/abs/1710.00925).

<div align="center">
<img src="conan-cruise.gif" /><br><br>
</div>
### Install

```shell
conda create -n deep-head-pose python=3.6
conda activate deep-head-pose 
pip install -r requirements.txt
```

### Datasets

[数据集下载网址](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)

下载300W-LP数据集以及AFLW2000-3D数据集来重现实验操作

### Train & Test

由于dataloader需要读取txt文件中的图片路径，先要通过代码提取出所有数据的路径，在更改好数据集路径后运行文件路径提取：

```shell
python code/dataset_300W.py
# Or
python code/dataset_AFLW.py
```

训练模型可以通过运行`bash code/train_<model>.sh`实现,注意要指定数据集所在的路径，或者通过命令行进行训练：

```shell
python code/train_hopenet.py --data_dir PATH_OF_DATASET --filename_list LIST_FILE --dataset TYPE_OF_DATASET --lr LEARN_RATE --alpha ALPHA --gpu GPU_ID --num_epochs NUM_EPOCH --batch_size BATCH_SIZE 
```

训练模型后需要进行测试,可以参考`code/test_<model>.sh`,注意提取的snapshot的路径,通过命令行进行测试:

```	shell
python test_hopenet.py --data_dir PATH_OF_DATASET --filename_list LIST_FILE --dataset TYPE_OF_DATASET --snapshot <SNAPSHOT_PATH>/<PKL_FILENAME>.pkl --gpu GPU_ID
```

使用dlib人脸检测在视频上进行测试（头部中心会跳动）:
```bash
python code/test_on_video_dlib.py --snapshot PATH_OF_SNAPSHOT --face_model PATH_OF_DLIB_MODEL --video PATH_OF_VIDEO --output_string STRING_TO_APPEND_TO_OUTPUT --n_frames N_OF_FRAMES_TO_PROCESS --fps FPS_OF_SOURCE_VIDEO
```
Face bounding box annotations should be in Dockerface format (n_frame x_min y_min x_max y_max confidence).

### Pre-trained models

官方的预训练模型的链接如下：

[300W-LP, alpha 1](https://drive.google.com/open?id=1EJPu2sOAwrfuamTitTkw2xJ2ipmMsmD3)

[300W-LP, alpha 2](https://drive.google.com/open?id=16OZdRULgUpceMKZV6U9PNFiigfjezsCY)

[300W-LP, alpha 1, robust to image quality](https://drive.google.com/open?id=1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR)

我们进行实验，下面将展示从头开始训练出来的最好的模型以及对上述官方模型中的[300W-LP, alpha 1](https://drive.google.com/open?id=1EJPu2sOAwrfuamTitTkw2xJ2ipmMsmD3)进行再训练优化后的最好结果：

[300W-LP，lr1e-5， alpha1，epoch5](https://drive.google.com/file/d/1Mek647SNH3YwOhOnCumBrDgVZpEXwdm2/view?usp=sharing)

[300W-LP，lr1e-6，alpha1，epoch5，pretrained](https://drive.google.com/file/d/1y6-gwulRVCSNuxtUn2IQ0a8E1ylt4qkh/view?usp=sharing)

### Cite


如果你应用Hopennet且觉得非常有效，请在使用处标明:

```
@InProceedings{Ruiz_2018_CVPR_Workshops,
author = {Ruiz, Nataniel and Chong, Eunji and Rehg, James M.},
title = {Fine-Grained Head Pose Estimation Without Keypoints},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2018}
}
```

*Nataniel Ruiz*, *Eunji Chong*, *James M. Rehg*

Georgia Institute of Technology
