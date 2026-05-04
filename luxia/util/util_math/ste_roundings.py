import torch
from luxia.types import round_t


class STE_Ceil(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        x_round = torch.ceil(input)
        return x_round

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Bypass gradient unchanged


class STE_Floor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        x_round = torch.floor(input)
        return x_round

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Bypass gradient unchanged


class STE_Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        x_round = torch.round(input)
        return x_round

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Bypass gradient unchanged


class STE_Trunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        x_round = torch.trunc(input)
        return x_round

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Bypass gradient unchanged


def apply_ste_rounding(x, round_type):

    if round_type == round_t.FLOOR:
        return STE_Floor.apply(x)
    elif round_type == round_t.CEIL:
        return STE_Ceil.apply(x)
    elif round_type == round_t.ROUND:
        return STE_Round.apply(x)
    elif round_type == round_t.TRUNC:
        return STE_Trunc.apply(x)
    else:
        raise ValueError("Invalid rounding type")
