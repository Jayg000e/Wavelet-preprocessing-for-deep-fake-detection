
## faceformer

本代码基于https://github.com/microsoft/Swin-Transformer

主要做了以下几点改动：

1.data/preprocess.py preprocess_real.sh preprocess_fake.sh三个文件利用离散小波变换预处理人脸数据， 生成对应的小波分量

2.如果选取小波分量加入输入通道，那么共会有21个输入通道而不是3个（详见报告），因此在main_face.py和config.py中对命令行输入和处理有
相应改动

3.舍弃了一切数据增强操作，因为我认为分类任务的数据增强方式不利于人脸的训练（详见报告），因此data/build.py中去除了很多
transform操作，同时改写了data/build.py中其中的数据IO，以便输入人脸数据

4.train.sh脚本和train_wavelet.sh脚本分别用于训练利用小波分量和不利用小波分量训练

## 模型表现

我们仅采用10000个真实人脸和10000个虚假人脸进行训练，在60000个真实人脸和60000个虚假人脸上进行测试，使用swin_transformer
的预训练模型swin-T 224x224,以下结果在25个epoch以内都能达到

1.不使用小波分量的情况下准确率为99.520%，AUC为99.987%

2.在使用小波分量的情况下准确率为99.828%,AUC为99.998%

##数据准备
1. mkdir fakeset 

2. 按照https://github.com/NVlabs/stylegan2
   中的指示配置环境，生成70000张虚假人脸
其中seeds选项改为0-69999，图片保存在fakeset/generated_image文件夹中，这大概需要一天， 
   取决于你使用的GPU
![img.png](img.png)
   
3.按照https://github.com/NVlabs/stylegan2 
中的指示下载FFHQ数据集中images1024x1024文件夹，对应
fakeset/images1024x1024文件夹

4.运行preprocess_real.sh,preprocess_fake.sh文件生成小波分量,这个过程大概需要一天，取决于你用的CPU


5.在进行上面几步后，你的fakeset文件夹下面应当有下面几个文件夹，每个文件夹都应当包含70000个数据文件
![img_1.png](img_1.png)

##开始训练和评估

1.按照https://github.com/microsoft/Swin-Transformer
中的提示配置环境，本代码还需要安装很少的几个包，运行时如果报告
环境缺失补充安装即可。（注意要和上述生成图片的环境完全隔离开）。代码在CUDA10.2 python3.7环境下能够成功运行，其他环境不作保证。

2.按照https://github.com/microsoft/Swin-Transformer
中的提示下载imagenet预训练的
swin_tiny_patch4_window7_224.pth文件

3，运行train.sh或train_wavelet.sh文件，取决于你是否想利用小波分量，如果你使用这两个文件进行了至少1个epoch的训练，但中间因为某些原因中断了
训练，只需要把脚本中的--swin_pretrained选项删除，然后继续运行这两个脚本即可。如果你所想使用的gpu数量不是脚本中的4个，把脚本中gpu数量作相应修改即可

##训练所需时间

由于我们仅仅使用20000张图像进行训练，且训练25个epoch以内，因此在4张2080ti的计算资源下，一天以内可以训练完，但由于测试使用120000张图像，
训练加测试的时间在4卡情况为1个星期以内








# Wavelet-preprocessing-for-deep-fake-detection.
