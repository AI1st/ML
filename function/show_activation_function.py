import torch
from Common.function.graph_plot import plot
import torch.nn.functional as F
from d2l import torch as d2l


# 可视化激活函数函数
def compute_error_from_linear(x, y):
    return torch.sqrt(F.mse_loss(x, y))


def plot_activation_function(model, sequence, deep, device):
    x_w = torch.linspace(-500, 500, 500, device=device)
    x_n = torch.linspace(-10, 10, 100, device=device)
    y_w = model(x_w).detach().cpu()
    y_n = model(x_n).detach().cpu()
    x_w = x_w.cpu()
    x_n = x_n.cpu()
    error_w = compute_error_from_linear(x_w,y_w)
    error_n = compute_error_from_linear(x_n,y_n)
    params = [str(round(i.detach().float().item(), 3)) for i in model.parameters()]
    plot(x_w, y_w, xlabel="input", ylabel="output",
         title=f"deep:{deep}, layer:{sequence}, error:{round(error_w.item(), 3)},\n p:{', '.join(params)}")
    plot(x_n, y_n, xlabel="input", ylabel="output",
         title=f"deep:{deep}, layer:{sequence}, error:{round(error_n.item(), 3)},\n p:{', '.join(params)}")


def show_activation(model, activation, deep=0, device=d2l.try_gpu()):
    i = 1
    for layer in model:
        if isinstance(layer, activation):
            plot_activation_function(layer, i, deep, device=device)
            i += 1
        else:
            try:
                show_activation(layer, activation, deep + 1)
            except Exception as e:
                pass
