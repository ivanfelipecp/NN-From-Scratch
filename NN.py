import abc
import torch
import numpy as np

class NN():
    def __init__(self, loss_function, learning_rate, net):
        self._LOSS_FUNCTION = loss_function
        self._LR = learning_rate

        self._NETWORK = self.build_network(net)

        self._FIRST_LAYER = 0
        self._LAST_LAYER = len(self._NETWORK) - 1
        self._PREDICT = None

    def predict(self, x):
        self._PREDICT = self._NETWORK[self._FIRST_LAYER].forward(x)
        print(self._PREDICT.shape)
    
    def build_network(self, net):
        temp = net[0]        
        for i in net[1:]:
            temp._SUCCESSOR = i
            i._PREDECCESSOR = temp
            temp = i
        return net

class Module(metaclass=abc.ABCMeta):
    def __init__(self):
        self._SUCCESSOR = None
        self._PREDECCESSOR = None
        self._FORWARD_RESULT = None
        self._BACKWARD_RESULT = None

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def backward(self):
        pass

    @abc.abstractmethod
    def update(self):
        pass

    @abc.abstractmethod
    def zero_gradient(self):
        pass    

class Lineal(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._INPUT_SIZE = input_size
        self._OUTPUT_SIZE = output_size
        self._W = None

        self.init_weights()

    def init_weights(self):
        #self._W = np.random.rand(self._INPUT_SIZE, self._OUTPUT_SIZE)*np.sqrt(1/(self._NETWORK._INPUT_SIZE+self._NETWORK._OUTPUT_SIZE))
        self._W = np.random.randn(self._INPUT_SIZE, self._OUTPUT_SIZE) / np.sqrt(self._INPUT_SIZE)

    def forward(self, x):
        #pass
        self._FORWARD_RESULT = np.dot(x, self._W)
        return self._SUCCESSOR.forward(self._FORWARD_RESULT) if self._SUCCESSOR else self._FORWARD_RESULT

    def backward(self):
        pass

    def update(self):
        pass

    def zero_gradient(self):
        pass

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #pass
        self._FORWARD_RESULT =  1 / (1 + np.exp(-x))
        return self._SUCCESSOR.forward(self._FORWARD_RESULT) if self._SUCCESSOR else self._FORWARD_RESULT

    def backward(self):
        pass

    def update(self):
        pass

    def zero_gradient(self):
        pass


def asd():
    return 1 if True else 0


"""
l = Lineal(784,10)
X = np.random.rand(1, 784)
z = l.forward(X)
print(z.shape)
#print(l._FORWARD_RESULT.shape)
"""
X = np.random.rand(15, 784)
layers = (
    Lineal(784,128),
    Sigmoid(),
    Lineal(128,10),
    Sigmoid()
)

# (input_size, output_size, loss_function, learning_rate, net):
nn = NN(None, 0.0085, layers)
nn.predict(X)
#print(layers[0]._W.shape)
#asd(1,2,3,4,5,6)
#a = tuple(i for i in range(1,10)[:-1])
#print(a)