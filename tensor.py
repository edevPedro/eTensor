import numpy as np

NDArray = np.ndarray


def _ensure_tensor(val):
  return val if isinstance(val, Tensor) else Tensor(val, requires_grad=False)


class Tensor:
  def __init__(
    self,
    data,
    *,
    device=None,
    dtype="float32",
    requires_grad=False,
    _op=None,
    _inputs=None,
  ):
    """
      Create a new Tensor
    """

    if isinstance(data, Tensor):
        if dtype is None:
            dtype = data.dtype
        self.data = data.numpy().astype(dtype)
    elif isinstance(data, np.ndarray):
        self.data = data.astype(dtype if dtype is not None else data.dtype)
    else:
        self.data = np.array(data, dtype=dtype if dtype is not None else "float32")

    self.grad = None
    self.requires_grad = requires_grad
    self._device = device if device else "cpu"
    self._op = _op
    self._inputs = _inputs if _inputs is not None else []

  @property
  def shape(self):
    """
      Returns the shape of the tensor.

      example:
        t = Tensor([1, 2, 3])
        print(t.shape)  # (3,)
    """
    return self.data.shape

  @property
  def dtype(self):
    """
      Returns the data type of the tensor.

      example:
        t = Tensor([1, 2, 3])
        print(t.dtype)  # float32

    """
    return self.data.dtype

  @property
  def ndim(self):
    """
      Returns the number of dimensions of the tensor.

      example:
        t = Tensor([1, 2, 3])
        print(t.ndim)  # 1
    """
    return self.data.ndim

  @property
  def size(self):
    """
      Returns the number of elements in the tensor.

      example:
        t = Tensor([1, 2, 3])
        print(t.size)  # 3
    """
    return self.data.size

  @property
  def device(self):
    """
      Returns the device the tensor is on.
      Device is basically where the tensor's data is stored. i.e. "cpu" or "cuda".

      example:
        t = Tensor([1, 2, 3])
        print(t.device)  # cpu
    """
    return self._device

  def numpy(self):
    """
      Returns a copy of the tensor's data as a NumPy array.
      example:
        t = Tensor([1, 2, 3])
        print(t.numpy())  # [1 2 3]
    """
    return self.data.copy()

  def __repr__(self):
    """
      Returns a string representation of the tensor.
      example:
        t = Tensor([1, 2, 3])
        print(t)  # Tensor([1, 2, 3], requires_grad=False)
    """
    return f"Tensor({self.data}, requires_grad={self.requires_grad})"

  def __str__(self):
    """
      Returns a string representation of the tensor.
      example:
        t = Tensor([1, 2, 3])
        print(t)  # Tensor([1, 2, 3])
    """
    return str(self.data)

  def detach(self):
    """
      Returns a new tensor with the same data but without gradient tracking.
      example:
        t = Tensor([1, 2, 3])
        t_detached = t.detach()
        print(t_detached)  # Tensor([1, 2, 3], requires_grad=False)
    """
    return Tensor(self.data, requires_grad=False)

  def backward(self):
    #todo
    pass
