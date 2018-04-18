# DDN
# DDN is a saliency model built on DenseNet
##Dependencies
- Caffe
- Python
- Linux

###Test your images
1. Download the project code
2. Download our pretrained model from Baidu Yun (https://pan.baidu.com/s/1psLBqqjGSmF0GFyQX9mzog)
3. Change the image path and save path in test.py
4. Run --python test.py

###Train your own model
1. Change the image path and ground-truth path in create_caffe_data.py
2. Run --python create_caffe_data.py
3. Chang the txt path(created by create_caffe_data.py) in DDN.prototxt
4. Run --python solve.py
