from micrograd.value import Value

def MSE(y_hat : list[float],y_pred : list[Value]) :
  return sum(((yout-ygt)**2 for ygt, yout in zip(y_hat, y_pred)), Value(0.0))
