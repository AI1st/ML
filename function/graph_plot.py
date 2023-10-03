import matplotlib.pyplot as plt


def plot(x, ys, labels=None, xlabel="", ylabel="", title=""):
    # 判断ys是否为容器类型变量(判断ys是否传入了多变量)
    if not isinstance(ys, (list, tuple, set, dict)):
        ys = [ys]
        labels = [labels]
    # 判断标签是否为空
    if labels is None:
        labels = [""] * len(ys)
    fig, ax = plt.subplots()
    for y, label in zip(ys, labels):
        ax.plot(x, y, label=label)
    ax.set_xlabel(xlabel)  # 设置x轴名称 x label
    ax.set_ylabel(ylabel)  # 设置y轴名称 y label
    ax.set_title(title)  # 设置图名为Simple Plot
    ax.legend()  # 自动检测要在图例中显示的元素，并且显示
