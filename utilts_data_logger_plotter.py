import numpy as np
import matplotlib.pyplot as plt
import time
import os
# 注意这里数组一行是一个整的数据，n行就代表时间为n,这就免得一维数组少一个维度的问题了

class DataLogger:
    def __init__(self, dt, save_dir="logs", max_len=1000000):
        """
        save_dir: 保存目录
        max_len: 预分配长度（避免频繁扩容）
        """
        self.dt = dt
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.max_len = max_len
        self.data = {}
        self.counter = 0
        self.add_signal("t",1)

    def add_signal(self, name, dim):
        """注册一个要记录的数据通道"""
        self.data[name] = np.zeros((self.max_len, dim))

    def log(self, name, value):
        """记录一次数据"""
        if name not in self.data:
            raise ValueError(f"Signal {name} not registered")

        self.data[name][self.counter] = value

    def step(self):
        """每次循环结束调用一次"""
        self.log("t",self.counter * self.dt)
        self.counter += 1
        if self.counter >= self.max_len:
            print("Logger buffer full!")
            self.counter = self.max_len - 1

    def get(self, name):
        """获取记录的数据"""
        return self.data[name][:self.counter]

    def save(self):
        """保存所有数据到 npy 文件"""
        for name, arr in self.data.items():
            np.save(os.path.join(self.save_dir, f"{name}.npy"), arr[:self.counter])
        print(f"Saved logs to {self.save_dir}")

    def plot(self, name):
        """简单画图"""
        sig = self.get(name)
        plt.figure()
        plt.plot(sig)
        plt.title(name)
        plt.grid()
        plt.show()


def plot_time_series(
    t,
    signals,
    labels=None,
    title="",
    xlabel="Time (s)",
    ylabel="Value",
    color=None,
    linewidth=2,
    figsize=(8, 4),
    legend_loc="best",
    grid=True,
    save_path=None,
    dpi=300,
    block=False
):
    
    signals = np.array(signals)

    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    n_signals = signals.shape[1]

    plt.figure(figsize=figsize)

    for i in range(n_signals):
        label = labels[i] if labels else None
        # plt.plot(t, signals[:, i], linewidth=linewidth, label=label)
        if color is None:
            plt.plot(t, signals[:, i], linewidth=linewidth, label=label)
        else:
            plt.plot(t, signals[:, i], color = color[i], linewidth=linewidth, label=label)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title( title, fontsize=14, fontweight="bold")

    if grid:
        plt.grid(True, linestyle="--", alpha=0.5)

    if labels:
        plt.legend(loc=legend_loc, fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi)   # ← dpi added

    plt.show(block=block)
    plt.pause(0.001)

def plot_subplots_time_series(
    t,
    signals_list,
    labels_list,
    titles_list,
    figsize=(10, 8),
    sharex=True,
    xlabel="Time (s)",
    ylabel_list=None,
    color = None,
    linewidth=2,
    save_path=None,
    dpi=300,
    block= False
):
    n = len(signals_list)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=sharex)
    if n == 1:
        axes = [axes]
    for i in range(n):
        sig = signals_list[i]
        if sig.ndim == 1:
            sig = sig.reshape(-1, 1)

        for j in range(sig.shape[1]):
            if color is None:
                axes[i].plot(t, sig[:, j], linewidth=linewidth, label=labels_list[i][j])
            else:
                axes[i].plot(t, sig[:, j], color = color[j], linewidth=linewidth, label=labels_list[i][j])

        axes[i].set_title(titles_list[i], fontsize=13, fontweight="bold")
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].legend()
        if(ylabel_list):
            axes[i].set_ylabel(ylabel_list[i], fontsize=13)

    axes[-1].set_xlabel(xlabel, fontsize=13)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi)   # ← dpi added

    plt.show(block=block)   

def plot_subplots_time_series_comparison(
    t,
    signals_list,
    labels_list,
    titles_list,
    figsize=(10, 8),
    sharex=True,
    xlabel="Time (s)",
    ylabel_list=None,
    color = None,
    linewidth=2,
    save_path=None,
    dpi=300,
    block= False
):
    n = len(signals_list)
    m1,n1 = (signals_list[0]).shape
    
    fig, axes = plt.subplots(n1, 1, figsize=figsize, sharex=sharex)

    if n1 == 1:
        axes = [axes]

    for i in range(n1):
        sig_list = []
        for p in range(n):
            sig_list.append(signals_list[p][:,i])
        sig = np.array(sig_list).transpose()
        if sig.ndim == 1:
            sig = sig.reshape(-1, 1)

        for j in range(sig.shape[1]):
            if color is None:
                axes[i].plot(t, sig[:, j], linewidth=linewidth, label=labels_list[i][j])
            else:
                axes[i].plot(t, sig[:, j], color = color[j], linewidth=linewidth, label=labels_list[i][j])

        axes[i].set_title(titles_list[i], fontsize=13, fontweight="bold")
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].legend()
        if(ylabel_list):
            axes[i].set_ylabel(ylabel_list[i], fontsize=13)

    axes[-1].set_xlabel(xlabel, fontsize=13)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi)   # ← dpi added

    plt.show(block=block)   


if __name__ == "__main__":
    path_prefix = "control_logs/"

    t = np.load(path_prefix+"t.npy")
    tau = np.load(path_prefix+"tau.npy")
    q_E_d = np.load(path_prefix+"q_E_d.npy")
    q_E_r = np.load(path_prefix+"q_E_r.npy")
    q_E   = np.load(path_prefix+"q_E.npy")

    eta1_E_d = np.load(path_prefix+"eta1_E_d.npy")
    eta1_E_r = np.load(path_prefix+"eta1_E_r.npy")
    eta1_E = np.load(path_prefix+"eta1_E.npy")

    
    eta_d = np.load(path_prefix+"eta_d.npy")
    eta_r = np.load(path_prefix+"eta_r.npy")
    eta = np.load(path_prefix+"eta.npy")


    plot_subplots_time_series_comparison(
    t,
    signals_list = [q_E_d,q_E_r,q_E],
    labels_list=[
        ["$q_{1,d}$","$q_{1,r}$","$q_1$"],
        ["$q_{2,d}$","$q_{2,r}$","$q_2$"],
        ["$q_{3,d}$","$q_{3,r}$","$q_3$"],
        ["$q_{0,d}$","$q_{0,r}$","$q_0$"]
    ],
    titles_list=["","","",""],
    figsize=(10, 8),
    sharex=True,
    xlabel="Time (s)",
    ylabel_list=["$q_1$","$q_2$","$q_3$","$q_0$"],
    save_path= path_prefix+"q_E.png",
    dpi=300,
    block= False
)
    plot_subplots_time_series_comparison(
    t,
    signals_list = [eta1_E_d,eta1_E_r,eta1_E],
    labels_list=[
        ["$\eta_{1,E,d}$","$\eta_{1,E,r}$","$\eta_{1,E}$"],
         ["$\eta_{2,E,d}$","$\eta_{2,E,r}$","$\eta_{2,E}$"],
         ["$\eta_{3,E,d}$","$\eta_{3,E,r}$","$\eta_{3,E}$"],
    ],
    titles_list=["","",""],
    figsize=(10, 8),
    sharex=True,
    xlabel="Time (s)",
    ylabel_list=["x (m)","y (m)","z (m)"] ,
    save_path=path_prefix+"eta1_E.png",
    dpi=300,
    block= False
)
    plot_subplots_time_series_comparison(
    t,
    signals_list = [eta_d,eta_r,eta],
    labels_list=[
        ["$x_d$","$x_r$","$x$"],
        ["$y_d$","$y_r$","$y$"],
        ["$z_d$","$z_r$","$z$"],
        ["$R_{x,d}$","$R_{x,r}$","$R_x$"],
        ["$R_{y,d}$","$R_{y,r}$","$R_y$"],
        ["$R_{z,d}$","$R_{z,r}$","$R_z$"],
    ],
    titles_list=["","","", "","",""],
    figsize=(10, 8),
    sharex=True,
    xlabel="Time (s)",
    ylabel_list=["x (m)","y (m)","z (m)","Rx (rad)","Ry (rad)","Rz (rad)"] ,
    save_path=path_prefix+"eta.png",
    dpi=300,
    block= False
)

    plot_time_series(
    t,
    signals= tau,
    labels=["X","Y","Z","K","M","N"],
    title="",
    xlabel="Time (s)",
    ylabel="tau (N/Nm)",
    linewidth=2,
    figsize=(8, 4),
    legend_loc="best",
    grid=True,
    save_path=path_prefix+"tau.png",
    dpi=300,
    block=True
    )