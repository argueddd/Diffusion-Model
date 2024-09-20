import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
df = pd.read_csv('../../time-series-forecasting/kqi_hour_datacell_nr-20240918.csv')

# 假设要取的列名是 'column_name'
data = df['ultraffic']

# 绘制折线图
plt.plot(data)
plt.title('Column Line Plot')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

# 显示图像
plt.show()