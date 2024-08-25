from collections import Counter

import numpy as np
from scipy.signal import convolve

from plot import plot_generate_result


def calculate_probability_distribution(signal, batch_size):
    prob_list = []
    signal_min = min(signal)
    signal_max = max(signal)
    interval = (signal_max - signal_min) / batch_size

    # 初始化计数器列表
    count_list = [0] * batch_size

    # 统计每个区间的频数
    for x in signal:
        index = min(int((x - signal_min) / interval), batch_size - 1)
        count_list[index] += 1

    # 计算每个区间的概率

    for i in range(batch_size):
        lower_bound = signal_min + i * interval
        prob_list.append([lower_bound, count_list[i] / len(signal)])

    return prob_list


def calculate_probabilities_distribution_by_freq(lst):
    # 统计每个元素出现的频率
    frequency_counter = Counter(lst)

    # 计算总元素数
    total_count = len(lst)

    # 计算每个元素的概率
    probability_distribution = {key: round(value / total_count, 5) for key, value in frequency_counter.items()}

    return probability_distribution


def find_duplicates(lst):
    return list(set([x for x in lst if lst.count(x) > 1]))


def generate_signal():
    dt = 0.001
    fm = 35
    L = 200
    t = np.arange(-L / 2, L / 2 + 1) * dt
    N = 500
    # 生成信号
    x = (1 - 2 * (np.pi * fm * t) ** 2) * np.exp(-(np.pi * fm * t) ** 2)
    # 添加反射系数
    r = (np.random.normal(0, 0.28, N).round(3)) ** 3
    # 信号卷积
    s = convolve(x, r)
    return x, r, s


def calculate_transfer_matrix(dic):
    n = len(dic.keys())
    prob_dis = np.array(list(dic.values())).reshape(1, -1)
    alpha = np.random.uniform(0.5, 0.8)
    ones = np.ones((n, 1))
    # 计算转移矩阵 P = alpha*I + (1-alpha)*ones*prob_dis
    transfer_matrix = alpha * np.eye(n) + (1 - alpha) * np.dot(ones, prob_dis)
    return transfer_matrix


def generate_new_ref(dic, P):
    ori_r = list(dic.keys())
    new_r = []
    for i in reflectance:
        position = ori_r.index(i)
        item_transfer_probability = P[position, :]
        sample = np.random.choice(ori_r, p=item_transfer_probability)
        new_r.append(sample)
    return new_r


if __name__ == '__main__':
    # 基本参数设置
    riker, reflectance, seismic = generate_signal()

    prob_dic = calculate_probabilities_distribution_by_freq(reflectance)
    transform_matrix = calculate_transfer_matrix(prob_dic)
    new_ref = generate_new_ref(prob_dic, transform_matrix)

    plot_generate_result(reflectance, new_ref)
