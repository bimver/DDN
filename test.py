import numpy as np
import sys
import caffe
import cv2
import matplotlib.pyplot as plt
import os
import sys
import time

def initNet(root_path='./',device_no=0):
    MODEL_FILE = root_path+'DDNdeploy.prototxt'
    PRETRAINED = root_path+'DDN.caffemodel'
    if device_no>=0:
        caffe.set_device(1)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Classifier(MODEL_FILE, PRETRAINED,caffe.TEST)
    return net                       

#process only one image, one can modify it to process a batch of images
def processImage(net,filename):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.array([ 104.00698793,  116.66876762,  122.67891434]))
    transformer.set_raw_scale('data', 255)  # images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # channels in BGR order instead of RGB
    img=caffe.io.load_image(filename)
    (H,W,C)=img.shape   #C=3
    if C==1:
        img[:,:,2]=img[:,:,1]
        img[:,:,3]=img[:,:,1]


    #process the image
    imgData=transformer.preprocess('data',img)
    net.blobs['data'].data[...] = imgData
    time_start= time.time()
    net.forward()
    time_end = time.time()
    print time_end - time_start,
    print "s"
    outmap = net.blobs['outmap'].data[0, 0, :, :]

    map_final = cv2.resize(outmap, (W, H))
    map_final -= map_final.min()
    map_final /= map_final.max()
    map_final = np.ceil(map_final * 255)
    return map_final


def vis_square(data):
    data = (data - data.min()) / (data.max() - data.min())
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                    (0, 1), (0, 1))  # add some space between filters
                   + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

        # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    #plt.imshow(data);
    #plt.axis('off')



net=initNet('./',0)
path = 'Test image path'
imgs = os.listdir(path)
for f_img in imgs:
    filename=path+f_img
    salmap = processImage(net, filename)
    mapname='Your save path'+f_img
    cv2.imwrite(mapname,salmap)
#feat = net.blobs['conv7'].data[0, :36]
#vis_square(feat)



