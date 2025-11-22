import numpy as np
from typing import Optional, Union, List, Callable
from src.utils import GradientBuffer


class Operation:
    def __init__(self,
                 inputs: List[np.ndarray],
                 output: np.ndarray,
                 gradients_to_update: List[Optional[Union[np.ndarray, None]]],
                 backward_operation: Callable):
        """
        Args:
            - inputs: operation inputs (List[np.ndarray])
            - outputs: operation output (Optional[Union[np.ndarray, List[np.ndarray]]])
            - gradients_to_update: parameter gradients if for parameter of network or None (numpy.ndarray, None)
            - backward_operation: backward function for nn/functional.py
        """
        self.inputs = inputs
        self.output = output
        self.gradients_to_update = gradients_to_update
        self.backward_operation = backward_operation

        self.i0_shp = self.inputs[0].shape
        self.i1_shp = None
        if len(self.inputs) > 1:
            self.i1_shp = self.inputs[1].shape
        self.bwd_op_name = self.backward_operation.__name__

    def __repr__(self):
        """
        Use this with print(operation) to help debug.
        """
        return (f"Operation [{self.i0_shp}, {self.i1_shp}, {self.output.shape}, {self.gradients_to_update}, {self.bwd_op_name}]")


class Autograd:
    def __init__(self):
        """
        A check to make sure you don't create more than 1 Autograd at a time.
        """
        if getattr(self.__class__, "_has_instance", False):
            raise RuntimeError("Cannot create more than 1 Autograd instance")
        self.__class__._has_instance = True

        self.gradient_buffer = GradientBuffer()
        self.operation_list = []

    def __del__(self):
        """
        Class destructor.
        """
        del self.gradient_buffer
        del self.operation_list
        self.__class__._has_instance = False

    def add_operation(self,
                      inputs: List[np.ndarray],
                      output: np.ndarray,
                      gradients_to_update: List[Optional[Union[np.ndarray, None]]],
                      backward_operation: Callable):
        """
        Adds operation to operation list and puts gradients in gradient buffer for tracking
        Args:
            - inputs: operation inputs as numpy arrays (numpy.ndarray)
            - outputs: operation output as a numpy array (numpy.ndarray)
            - gradients_to_update: parameter gradients if for parameter of network or None (numpy.ndarray, None)
            - backward_operation: backward function for nn/functional.py
        Returns:
            No return required
        """
        if len(inputs) != len(gradients_to_update):
            raise Exception("Number of inputs must match the number of gradients to update!")

        for input in inputs:
            self.gradient_buffer.add_spot(input)

        op = Operation(inputs, output, gradients_to_update, backward_operation)
        self.operation_list.append(op)



    def backward(self, divergence):
        """
        The backpropagation through the self.operation_list with a given divergence.
        This function should automatically update gradients of parameters by checking
        the gradients_to_update.
        Args:
            - divergence: loss value (float/double/int/long)
        Returns:
            No return required
        """
        for operation in reversed(self.operation_list):
            if operation == self.operation_list[-1]:
                grad = divergence
            else:
                grad = self.gradient_buffer.get_param(operation.output)

            grads = operation.backward_operation(grad, *operation.inputs)

            for i, inp in enumerate(operation.inputs):
                g = grads[i]
                if g is None:
                    continue

                if self.gradient_buffer.is_in_memory(inp):
                    self.gradient_buffer.update_param(inp, g)

                if operation.gradients_to_update[i] is not None:
                    operation.gradients_to_update[i] += g



    def zero_grad(self):
        """
        Resets gradient buffer and operations list.
        """
        self.gradient_buffer.clear()
        self.operation_list = []
