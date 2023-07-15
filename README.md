
# Wavelet-preprocessing-for-deep-fake-detection

This code is based on https://github.com/microsoft/Swin-Transformer and has made the following changes:

1.The three files data/preprocess.py, preprocess_real.sh, and preprocess_fake.sh use discrete wavelet transform to preprocess face data and generate corresponding wavelet components.

2.If the wavelet component is selected to be added to the input channel, there will be a total of 21 input channels instead of 3 (see the report for details). Therefore, there are corresponding changes in the command line input and processing in main_face.py and config.py.

3.All data augmentation operations have been abandoned because I believe that the data augmentation methods for classification tasks are not conducive to face training (see the report for details). Therefore, many transform operations have been removed in data/build.py, and the data IO in data/build.py has been rewritten to facilitate the input of face data.

4.The train.sh script and train_wavelet.sh script are used to train with and without wavelet components, respectively.

## Performance

We only used 10,000 real faces and 10,000 fake faces for training, and tested on 60,000 real faces and 60,000 fake faces using the pre-trained model of swin_transformer swin-T 224x224. The following results can be achieved within 25 epochs:

1.Without using wavelet components, the accuracy is 99.520%, and the AUC is 99.987%.

2.When using wavelet components, the accuracy is 99.828%, and the AUC is 99.998%.

## clone apex

Enter the wavelet_faceformer folder and clone the apex repository. 

## data preparation

To generate fake faces and wavelet components, you can follow these steps:

1.Create a fakeset folder by running mkdir fakeset.

2.Follow the instructions on https://github.com/NVlabs/stylegan2 to set up the environment and generate 70,000 fake faces. Change the seeds option to 0-69999, and save the images in the fakeset/generated_image folder. This process may take about a day, depending on the GPU you are using.

3.Follow the instructions on https://github.com/NVlabs/stylegan2 to download the images1024x1024 folder from the FFHQ dataset, corresponding to the fakeset/images1024x1024 folder.

4.Run the preprocess_real.sh and preprocess_fake.sh files to generate wavelet components. This process may take about a day, depending on the CPU you are using.

5.After completing the above steps, your fakeset folder should contain several subfolders, each containing 70,000 data files.


## Training and evaluation

To start training and evaluation, you can follow these steps:

1.Follow the instructions on https://github.com/microsoft/Swin-Transformer to set up the environment. This code also requires the installation of a few additional packages. If there are any missing environment dependencies reported during runtime, you can supplement the installation. (Note that the environment should be completely isolated from the environment used to generate images above). The code can run successfully in a CUDA10.2 python3.7 environment, but other environments are not guaranteed.

2.Follow the instructions on https://github.com/microsoft/Swin-Transformer to download the swin_tiny_patch4_window7_224.pth file pre-trained on ImageNet.

3.Run the train.sh or train_wavelet.sh file, depending on whether you want to use wavelet components. If you have trained for at least 1 epoch using these two files but interrupted the training for some reason, just delete the --swin_pretrained option in the script and continue running these two scripts. If the number of GPUs you want to use is not 4 as specified in the script, you can make corresponding changes to the number of GPUs in the script.

## Training Time

Since we only use 20,000 images for training and train within 25 epochs, it can be completed within one day with the computing resources of 4 2080ti GPUs. However, since the test uses 120,000 images, the time for training and testing is within one week under the condition of 4 GPUs.








