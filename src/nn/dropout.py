import numpy as np

class Dropout(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):

        if train:
   

            self.mask = np.random.binomial(1, 1 - self.p, size=x.shape) 
            x = x * self.mask / (1 - self.p)  # Scale the output by 1/(1-p) to maintain the expected value
            
            return x
        else:
            return x
		
    def backward(self, delta):

        return delta * self.mask