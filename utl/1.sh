tensorflowjs_converter --input_format keras --output_format=tfjs_graph_model mobilemodel.h5 ../static/models/mobile/
tensorflowjs_converter --input_format keras --output_format=tfjs_graph_model vgg16mode.h5 ../static/models/vgg16/
tensorflowjs_converter --input_format keras --output_format=tfjs_graph_model resnet50model.h5 ../static/models/resnet50/

