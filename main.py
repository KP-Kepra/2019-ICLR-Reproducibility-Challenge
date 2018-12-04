from networks import *
from tools import *

import trainer

# MiniAlexNet epoch : 10
# MLP3 epoch : 50
AllenNLP

# LeNet Training
batch_list = [128]
trainer.train('lenet', 'mnist', batch_list, 20, 'adadelta')

# MiniALexNet Training
batch_list = [500, 250, 100, 32, 16, 8, 4, 2]
trainer.train('minialexnet', 'cifar10', batch_list, 10, 'sgd', False)

# MLP3 Training
batch_list = [16]
trainer.train('mlp3', 'cifar10', batch_list, 50, 'sgd', True)
