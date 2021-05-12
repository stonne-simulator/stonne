from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List, Optional, Tuple

from torch import Tensor
from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F

from ..common_types import _size_any_t, _size_1_t, _size_2_t, _size_3_t, _ratio_3_t, _ratio_2_t

class _SimulatedPoolNd(Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation',
                     'return_indices', 'ceil_mode', 'path_to_arch_file', 'path_to_tile']
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    return_indices: bool
    ceil_mode: bool
    path_to_arch_file: str
    path_to_tile: str

    def __init__(self, kernel_size: _size_any_t, path_to_arch_file: str, path_to_tile: str, stride: Optional[_size_any_t] = None,
            padding: _size_any_t = 0, dilation: _size_any_t = 1, 
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super(_SimulatedPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.path_to_arch_file = path_to_arch_file
        self.path_to_tile = path_to_tile
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, path_to_arch_file={path_to_arch_file}, path_to_tile={path_to_tile}, stride={stride}, padding={padding}' \
            ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

class SimulatedMaxPool2d(_SimulatedPoolNd):
    r"""Applies a 2D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool2d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool2d((3, 2), stride=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t
    def __init__(self, kernel_size: _size_2_t, path_to_arch_file: str = '', path_to_tile: str = '', stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, return_indices: bool = False, ceil_mode: bool = False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SimulatedMaxPool2d, self).__init__(kernel_size, path_to_arch_file, path_to_tile, stride, padding, dilation, False, False)
    def forward(self, input: Tensor) -> Tensor:
        import torch_stonne
        return torch_stonne.simulated_max_pool_forward(self.__class__.__name__, input, self.kernel_size, self.stride, self.padding, self.dilation, self.path_to_arch_file, self.path_to_tile)
