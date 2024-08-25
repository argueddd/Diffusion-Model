import numpy as np

# 元素列表
elements = ['a', 'b', 'c', 'd']

# 概率分布，注意概率之和必须等于1
probabilities = [0.1, 0.3, 0.4, 0.2]

# 根据概率分布随机抽取一个元素
sample = np.random.choice(elements, p=probabilities)

print("Randomly chosen element:", sample)
