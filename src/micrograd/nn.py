import random
from micrograd.value import Value

class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0.0

  def parameters(self):
    return []


class Neuron(Module) :
  def __init__(self, n_inputs : int) -> None:
    self.w = [Value(random.uniform(-1,1)) for _ in range(n_inputs)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self, x) :
    # w*x +b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = act.tanh()

    return out

  def parameters(self):
    return self.w + [self.b]

class Layer(Module):
  def __init__(self, n_inputs, n_outputs):
    self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP(Module):
  def __init__(self, n_inputs, n_outs : list[int]) -> None:
    size = [n_inputs] + n_outs
    self.layers = [Layer(size[i], size[i+1]) for i in range(len(n_outs))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
