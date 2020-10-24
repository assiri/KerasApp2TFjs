

import keras 
import numpy as np 

from keras.applications import vgg16, inception_v3, resnet50, mobilenet 


#Load the Inception_V3 model 
#inception_model = inception_v3.InceptionV3(weights = 'imagenet') 

#Load the ResNet50 model 
resnet50model = resnet50.ResNet50(weights = 'imagenet') 
resnet50model.save('resnet50model.h5')


mobilenet_model = mobilenet.MobileNet(weights = 'imagenet')
mobilenet_model.save('mobilemodel.h5')

vgg16model = vgg16.VGG16(weights = 'imagenet') 
vgg16model.save("vgg16mode.h5")



# tensorflowjs_converter --input_format keras --output_format=tfjs_graph_model mobilemodel.h5 modeljs/mobile/
# tensorflowjs_converter --input_format keras --output_format=tfjs_graph_model vgg16mode.h5 modeljs/vgg16/
# tensorflowjs_converter --input_format keras --output_format=tfjs_graph_model resnet50model.h5 modeljs/resnet50/
