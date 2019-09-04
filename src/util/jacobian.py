import torch
import math
from torch.autograd.gradcheck import zero_gradients


def compute_jacobian(inputs, output):
    """
	inputs: Size (e.g. Depth X Width X Height)
	:param output: Classes
	:return: jacobian: Classes X Size
	"""
    assert inputs.requires_grad

    num_classes = output.size(0)

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)
