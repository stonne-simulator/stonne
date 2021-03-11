import math

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from .. import functional as F
from .. import init
from .module import Module
#import torch_stonne

class SimulatedLinear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.SimulatedLinear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features', 'path_to_arch_file', 'path_to_tile', 'sparsity_ratio']
    in_features: int
    out_features: int
    path_to_arch_file: str
    path_to_tile: str
    sparsity_ratio: float
    weight: Tensor


    def __init__(self, in_features: int, out_features: int, path_to_arch_file: str, path_to_tile: str, sparsity_ratio: float,  bias: bool = True) -> None:
        super(SimulatedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.path_to_arch_file = path_to_arch_file
        self.path_to_tile = path_to_tile
        self.sparsity_ratio = sparsity_ratio
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        import torch_stonne
        output =  torch_stonne.simulated_linear_forward(self.__class__.__name__, input, self.weight, self.path_to_arch_file, self.path_to_tile, self.sparsity_ratio, True) # The last true is to transpose the matrices 
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
