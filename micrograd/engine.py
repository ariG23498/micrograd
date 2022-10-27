from typing import Tuple, Union


class Value(object):
    """
    This is similar to the Node class in autograd. We need to wrap the
    raw data into a class that will store metadata to help in automatic
    differentiation.

    Args:
        data (float): The data for the Value node.
        _children (Tuple): The children of this node.
    """

    def __init__(self, data: float, _children: Tuple = ()):
        # The raw data for the Value node.
        self.data = data

        # The partial gradient of the last node with respect to this
        # node. This is also termed as the global gradient.
        # Gradient 0 means that there is no effect of the change
        # of the last node with respect to this node. On
        # initialization it is assumed that all the variables have no
        # effect on the entire architecture.
        self.grad = 0.0

        # The function that derives the gradient of the children nodes
        # of this current node. It is easier this way, because each node
        # is built from children nodes and an operation. Upon back-propagation
        # the current node can easily fill in the gradients of the children.
        # Note: The global gradient is the multiplication of the local gradeint
        # and the flowing gradient from the parent.
        self._backward = lambda: None

        # Define the children of this node.
        self._prev = set(_children)

    def __repr__(self):
        # This is the string representation of the Value node.
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: Union["Value", float]) -> "Value":
        """
        The addition operation for the Value class.
        Args:
            other (float): The other value to add to this one.
        Usage:
            >>> x = Value(2)
            >>> y = Value(3)
            >>> z = x + y
            >>> z.data
            5
        """
        # If the other value is not a Value, then we need to wrap it.
        other = other if isinstance(other, Value) else Value(other)

        # Create a new Value node that will be the output of the addition.
        out = Value(data=self.data + other.data, _children=(self, other))

        def _backward():
            # Local gradient:
            # x = a + b
            # dx/da = 1
            # dx/db = 1
            # Global gradient with chain rule:
            # dy/da = dy/dx . dx/da = dy/dx . 1
            # dy/db = dy/dx . dx/db = dy/dx . 1
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0

        # Set the backward function on the output node.
        out._backward = _backward
        return out

    def __radd__(self, other):
        """
        Reverse addition operation for the Value class.
        Args:
            other (float): The other value to add to this one.
        Usage:
            >>> x = Value(2)
            >>> y = Value(3)
            >>> z = y + x
            >>> z.data
            5
        """
        # This is the same as adding. We can reuse the __add__ method.
        return self + other

    def __mul__(self, other: Union["Value", float]) -> "Value":
        """
        The multiplication operation for the Value class.
        Args:
            other (float): The other value to multiply to this one.
        Usage:
            >>> x = Value(2)
            >>> y = Value(3)
            >>> z = x * y
            >>> z.data
            6
        """
        # If the other value is not a Value, then we need to wrap it.
        other = other if isinstance(other, Value) else Value(other)

        # Create a new Value node that will be the output of
        # the multiplication.
        out = Value(data=self.data * other.data, _children=(self, other))

        def _backward():
            # Local gradient:
            # x = a * b
            # dx/da = b
            # dx/db = a
            # Global gradient with chain rule:
            # dy/da = dy/dx . dx/da = dy/dx . b
            # dy/db = dy/dx . dx/db = dy/dx . a
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        # Set the backward function on the output node.
        out._backward = _backward
        return out

    def __rmul__(self, other):
        """
        Reverse multiplication operation for the Value class.
        Args:
            other (float): The other value to multiply to this one.
        Usage:
            >>> x = Value(2)
            >>> y = Value(3)
            >>> z = y * x
            >>> z.data
            6
        """
        # This is the same as multiplying. We can reuse the __mul__ method.
        return self * other

    def __neg__(self):
        """
        Negation operation for the Value class.
        Usage:
            >>> x = Value(2)
            >>> z = -x
            >>> z.data
            -2
        """
        # This is the same as multiplying by -1. We can reuse the
        # __mul__ method.
        return self * -1

    def __sub__(self, other):
        """
        Subtraction operation for the Value class.
        Args:
            other (float): The other value to subtract to this one.
        Usage:
            >>> x = Value(2)
            >>> y = Value(3)
            >>> z = x - y
            >>> z.data
            -1
        """
        # This is the same as adding the negative of the other value.
        # We can reuse the __add__ and the __neg__ methods.
        return self + (-other)

    def __rsub__(self, other):
        """
        Reverse subtraction operation for the Value class.
        Args:
            other (float): The other value to subtract to this one.
        Usage:
            >>> x = Value(2)
            >>> y = Value(3)
            >>> z = y - x
            >>> z.data
            1
        """
        # This is the same as subtracting. We can reuse the __sub__ method.
        return other - self

    def __pow__(self, other):
        """
        The power operation for the Value class.
        Args:
            other (float): The other value to raise this one to.
        Usage:
            >>> x = Value(2)
            >>> z = x ** 2.0
            >>> z.data
            4
        """
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"

        # Create a new Value node that will be the output of the power.
        out = Value(data=self.data ** other, _children=(self,))

        def _backward():
            # Local gradient:
            # x = a ** b
            # dx/da = b * a ** (b - 1)
            # Global gradient:
            # dy/da = dy/dx . dx/da = dy/dx . b * a ** (b - 1)
            self.grad += out.grad * (other * self.data ** (other - 1))

        # Set the backward function on the output node.
        out._backward = _backward
        return out

    def __truediv__(self, other):
        """
        Division operation for the Value class.
        Args:
            other (float): The other value to divide to this one.
        Usage:
            >>> x = Value(10)
            >>> y = Value(5)
            >>> z = x / y
            >>> z.data
            2
        """
        # Use the __pow__ method to implement division.
        return self * other ** -1

    def __rtruediv__(self, other):
        """
        Reverse division operation for the Value class.
        Args:
            other (float): The other value to divide to this one.
        Usage:
            >>> x = Value(10)
            >>> y = Value(5)
            >>> z = y / x
            >>> z.data
            0.5
        """
        # Use the __pow__ method to implement division.
        return other * self ** -1

    def relu(self):
        """
        The relu activation function.
        Usage:
            >>> x = Value(-2)
            >>> y = x.relu()
            >>> y.data
            0
        """
        out = Value(data=0 if self.data < 0 else self.data, _children=(self,))

        def _backward():
            # Local gradient:
            # x = relu(a)
            # dx/da = 0 if a < 0 else 1
            # Global gradient:
            # dy/da = dy/dx . dx/da = dy/dx . (0 if a < 0 else 1)
            self.grad += out.grad * (out.data > 0)

        # Set the backward function on the output node.
        out._backward = _backward
        return out

    def backward(self):
        """
        The backward pass of the backward propagation algorithm.
        Usage:
            >>> x = Value(2)
            >>> y = Value(3)
            >>> z = x * y
            >>> z.backward()
            >>> x.grad
            3
            >>> y.grad
            2
        """
        # build the topological sorted graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        # call the `build_topo` method on self
        build_topo(self)

        # go one variable at a time and apply the chain rule
        # to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
