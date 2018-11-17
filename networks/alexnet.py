'''
AlexNet Architecture

0. Input Layer         : 3 channels, 227x227
1. Convolutional Layer : 96 feature maps, 55x55 (kernel 11x11, stride 4)
2. MaxPool Layer       : 96 feature maps, 27x27 (kernel 3x3, stride 2)
3. Convolutional Layer : 256 feature maps, 27x27 (kernel 5x5, same padding)
4. MaxPool Layer       : 256 feature maps, 13x13 (kernel 3x3, stride 2)
5. Convolutional Layer : 384 feature maps, 13x13 (kernel 3x3, same padding)
6. MaxPool Layer       : 256 feature maps, 6x6 (kernel 3x3)
7. FC Layer            : 4096 nodes
8. FC Layer            : 4096 nodes
9. Output Layer        : 1000 classes (Softmax)
'''

# class Alex