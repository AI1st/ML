import torch
from graph_plot import plot


# 可视化激活函数函数
def plot_activation_function(model, sequence, deep):
    x_w = torch.linspace(-500, 500, 500)
    x_n = torch.linspace(-10, 10, 100)
    y_w = model(x_w).detach().numpy()
    y_n = model(x_n).detach().numpy()
    plot(x_w, y_w, xlabel="input", ylabel="output", title=f"deep:{deep}, layer:{sequence}  wide range output")
    plot(x_n, y_n, xlabel="input", ylabel="output", title=f"deep:{deep}, layer:{sequence}  narrow range output")


def show_activation(model, activation, deep=0):
    i = 1
    for layer in model:
        if isinstance(layer, activation):
            print(layer)
            plot_activation_function(layer, i, deep)
            i += 1
        else:
            try:
                show_activation(layer, activation, deep + 1)
            except:
                pass
