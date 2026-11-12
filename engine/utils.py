from engine.engine import Value
from graphviz import Digraph

def trace(root):
  nodes, edges = set(), set()

  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child,v))
        build(child)
    build(root)
  return nodes, edges

def draw_graph(root):
  dot = Digraph(format = 'svg', graph_attr = {'rankdir' : 'LR'}) # left to right ...

  nodes , edges = trace(root)

  for n in nodes :
    uid = str(id(n))

    dot.node(name =uid, label = "{ %s | %.3f | grad %.4f}" % (n._label, n.data, n.grad), shape = 'record') # create a rectangular node for ti
    if n.op: #create edges for the nodes and its results
      dot.node(name = uid + n._op , label = n._op)
      dot.edge(uid + n._op, uid)
  for n1, n2 in edges:
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot
