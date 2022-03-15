import numpy as np
import cv2
import os
import argparse
import pickle
from pywt import dwt2, idwt2,wavedec2

def transform_one(root,dst,img_path,transform):

    #get destination path name
    idx=os.path.basename(img_path[:-4])
    dst_path=os.path.join(dst,idx+transform+'.pkl')

    if os.path.exists(dst_path):
        if os.path.getsize(dst_path)==0:
            print(dst_path)
        else:
            return

    #read img
    img=cv2.imread(img_path).astype(np.float32)

    #do discreate cosine transform or wavelet transform
    if transform=='dct':
        result=np.stack(cv2.dct(img[:,:,i]) for i in range(3))
    elif transform=='wavelet':
        result=[list(wavedec2(img[:,:,i], 'haar',level=2)) for i in range(3)]

    #save result
    with open(dst_path, 'wb') as f:
        pickle.dump(result,f)

def transform_all(root,dst,transform):
    if not os.path.exists(dst):
        os.makedirs(dst)
    os.chdir(root)
    for file in os.listdir(root):
        if file.endswith('.png'):
            transform_one(root,dst,file,transform=transform)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--root', type=str)
    parser.add_argument('--dst',type=str)
    parser.add_argument('--transform',type=str)
    args = parser.parse_args()
    transform_all(args.root,args.dst,args.transform)

