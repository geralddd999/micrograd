from micrograd.value import Value
from micrograd.nn import Module


class Optimizer:
  def __init__(self, lr : float , parameters):
    self.lr = lr
    self.parameters = parameters
  def step(self):
    for p in self.parameters:
      p.data -= self.lr * p.grad

  def zero_grad(self):
    for p in self.parameters :
      p.grad = 0.0
