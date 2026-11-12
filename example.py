from micrograd.value import Value

from micrograd import nn
from src.micrograd.nn import MLP
from src.micrograd.optimizer import Optimizer
from src.micrograd.losses import MSE
from src.micrograd.utils import draw_graph

if __name__ == "__main__":

  # render an example that calculates the gradients:
  n = nn.Neuron(2)
  x = [Value(1.0), Value(-2.0)]
  y = n(x)
  y.backward()
  draw_graph(y).render()
  xs = [
    [2.0 , 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
  ]

  ys = [1.0, -1.0, -1.0, 1.0]


  n = MLP(3, [4,4,1])
  ypred = lambda : [n(x) for x in xs]
  optimizer = Optimizer(lr = 0.05, parameters = n.parameters())

  for k in range(100):
    # forward pass
    preds = ypred()
    loss = MSE(ys, preds)

    optimizer.zero_grad()
    #backward pass
    loss.backward()
    optimizer.step()

    print(f"step {k} loss : {loss.data}")
    print(f"preds : {[p.data for p in preds]}")
