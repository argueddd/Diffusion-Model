from matplotlib import pyplot as plt


def plot_srx(s, r, x):
    # riker子波
    plt.figure(figsize=(15, 10))
    # 原始信号
    plt.subplot(3, 1, 1)
    plt.plot(x, x)
    plt.title("Original Signal (w_R)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    # 反射系数
    plt.subplot(3, 1, 2)
    plt.plot(r)
    plt.title("Noise Signal (r)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    # 地震记录
    plt.subplot(3, 1, 3)
    plt.plot(s)
    plt.title("Convolved Signal (w1)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def plot_generate_result(old, new):
    # riker子波
    plt.figure(figsize=(15, 10))
    # 原始信号
    plt.subplot(2, 1, 1)
    plt.plot(old)
    plt.title("Original Signal (R)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.ylim(-1, 1)
    # 反射系数
    plt.subplot(2, 1, 2)
    plt.plot(new)
    plt.title("Generate Signal (R1)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.show()