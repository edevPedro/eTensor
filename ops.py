import numpy as np
from .tensor import NDArray, Tensor

class Function:
  def __call__(self, *inputs):
    requires_grad = any(t.requires_grad for t in inputs)
    input_data = [t.data for t in inputs]
    output_data = self.forward(*input_data)
    output_tensor = Tensor(output_data, requires_grad=requires_grad)
    if requires_grad:
      output_tensor._op = self
      output_tensor._inputs = inputs
    return output_tensor

  def forward(self, *args):
    """
      Perform the forward pass of the operation.
      It:
        - Compute new values.
        - Wrap those values inside Tensor.
    """
    raise NotImplementedError()

  def backward(self, *args):
    """
      Calculate the gradient of the function.
      Args:
          out_grad: upstream gradient flowing from output to input
          node: Value object holding inputs from forward pass
    """
    raise NotImplementedError()

class Add(Function):
  """
    Element-wise addition of two tensors.
    a + b
  """

  def forward(self, *args):
    a, b = args
    return a+b
  def backward(self, *args):
    out_grad, node = args
    return out_grad, out_grad

def add(a, b):
  return Add()(a, b)  #`__call__`

class Mul(Function):
  """
    Element-wise multiplication of two tensors.
    a * b
  """
  def forward(self, *args):
    a, b = args
    return a * b

  def backward(self, *args):
    out_grad, node = args
    a, b = node._inputs
    return out_grad * b, out_grad * a

def mul(a, b):
  return Mul()(a, b)

class Sub(Function):
  """
    Element-wise subtraction of two tensors.
    a - b
  """
  def forward(self, *args):
    a, b = args
    return a - b

  def backward(self, *args):
    out_grad, node = args
    a, b = node._inputs
    return out_grad, -out_grad

def sub(a, b):
  return Sub()(a, b)

class Div(Function):
  """
    Element-wise division of two tensors.
    a / b
  """
  def forward(self, *args):
    a, b = args
    return a / b

  def backward(self, *args):
    out_grad, node = args
    a, b = node._inputs
    return out_grad / b, -out_grad * a / b

def div(a, b):
  return Div()(a, b)

class Pow(Function):
  """
    Element-wise power of two tensors.
    a ** b
  """
  def forward(self, *args):
    a, b = args
    return a ** b

  def backward(self, *args):
    out_grad, node = args
    a, b = node._inputs
    return out_grad * b * a ** (b - 1),

class Transpose(Function):
  """
    Transpose of a tensor.
    np.transpose(a)

    In a simplified description, the transpose of a tensor is obtained by swapping its rows and columns.
  """
  def forward(self, *args):
    a = args[0]
    return np.transpose(a)

  def backward(self, *args):
    out_grad, node = args
    return np.transpose(out_grad),

class Reshape(Function):
  """
    Reshape of a tensor.
    np.reshape(a, shape)
    In a simplified description, the reshape of a tensor is obtained by changing its dimensions while keeping the total number of elements constant.
  """
  def forward(self, *args):
    a = args[0]
    shape = args[1]
    return np.reshape(a, shape)

  def backward(self, *args):
    out_grad, node = args
    return np.reshape(out_grad, node._inputs[0].shape),

class BroadcastTo(Function):
  """
    Broadcast a tensor to a new shape.
    np.broadcast_to(a, shape)

    In a simplified description, the broadcast of a tensor is obtained by repeating its elements to match the new shape.
  """
  def forward(self, *args):
    a = args[0]
    shape = args[1]
    return np.broadcast_to(a, shape)

  def backward(self, *args):
    out_grad, node = args

    original_shape = node._inputs[0].data.shape
    expanded_shape = out_grad.shape
    axis = tuple(i for i, (o, e) in enumerate(zip(original_shape, expanded_shape)) if o != e)

    return np.sum(out_grad, axis=axis),
