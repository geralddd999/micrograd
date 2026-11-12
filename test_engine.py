from src.micrograd.value import Value

def test_add():
  a = Value(2.0)
  b = Value(-3.0)

  d = a+b;
  assert(d.data == -1.0)

if __name__ == "__main__":
  test_add()
