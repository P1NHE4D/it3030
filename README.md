# IT3030 - Deep Learning - Project 01

Custom implementation of fully-connected neural networks.
The configs located in `configs` can be used as a template
for constructing custom networks. 

# Configuration Parameters
|Section | Parameter | Type | Value range | Description |
| ------ | --------- | ---- | ------ | ----------- |
| image_viewer | enabled | Boolean | - | Enables the image viewer |
| image_viewer | set | String | train, val, test | Selects the image set for the image viewer |
| image_viewer | plot_count | Int | 0 - N | Number of images to be plotted |
| image_viewer | random | Boolean | - | If true, images to be plotted are randomly selected |
| globals | lr | Float | - | learning rate |
| globals | epochs | Int | 0 - N | number of epochs |
| globals | loss | String | cross_entropy, mse | Loss function |
| globals | reg_type | String | l1, l2, none | Regularization type |
| globals | reg_rate | Float | - | Regularization penalty |
| globals | visualize | Boolean | - | If true, loss is plotted after training |
| globals | verbose | Boolean | - | If true, input, target value, network output, and instance loss is printed to the console |
| layers | type | String | flatten, dense, softmax | Layer type |
| layers | act | String | relu, tanh, sigmoid, linear | Activation function for the layer |
| layers | units | Int | 1 - N | Number of units in the layer |
| layers | wr | List[Float] | - | Weight initialization range |
| dataset | split_ratio | List[Float] | - | Train, val, test split ratio. Must sum up to 1. |
| dataset | img_dims | Int | 10 - 50 | Image dimensions |
| dataset | width_range | List[Int] | - | Width range of object to be drawn in image |
| dataset | height_range | List[Int] | - | Height range of object to be drawn in image |
| dataset | img_noise | Float | - | Regulates the amount of noise in the image |
| dataset | flatten | Boolean | - | If true, returns image as vector |
| dataset | size | Int | 1 - N | Size of the dataset | 
| dataset | centred | Boolean | - | If true, objects in image are centred |
| dataset | normalise | Boolean | - | If true, image is normalised |
