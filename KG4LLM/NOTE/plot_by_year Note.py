# ===== 导入绘图库 =====
from matplotlib import pyplot as plt
# 导入matplotlib的pyplot模块，用于创建各种图表

plt.style.use('seaborn')
# 设置matplotlib的绘图风格为seaborn，提供更美观的默认样式
# seaborn风格特点：更柔和的颜色，更清爽的背景

import seaborn
# 导入seaborn库，用于高级统计图表绘制
# seaborn.set(rc={'axes.facecolor':'gainsboro'})
# （注释掉）设置坐标轴背景色为浅灰色

# ===== 定义颜色方案 =====
color1 = "#ec661f"  # 橙色，用于DALK方法的折线
color2 = "#038355"  # 绿色，用于无自感知检索方法的折线  
color3 = '#9BB8F2'  # 蓝色，用于柱状图（三元组数量）

# ===== 定义数据 =====
years = [2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
# X轴：年份数据，表示知识图谱演化的时间轴

ours = [68.1,69.0,70.2,68.4,69.7,70.3,70.3,71.8,71.2,72.6]
# DALK方法在各年份的准确率（%）
# 显示DALK方法随时间的性能变化

no_self = [69.0,69.4,70.0,70.0,70.1,69.9,70.2,68.0,69.8,70.6]
# 无自感知知识检索方法的准确率（%）
# 用于对比展示自感知检索的效果

baseline = [67.1 for _ in years]
# 基线方法的准确率，所有年份都是67.1%
# 使用列表推导式创建常数列表：[67.1, 67.1, 67.1, ...]

count = [5661,6953,6349,7136,6587,5365,4953,5071,3309,3201]
# 每年新增的知识图谱三元组数量

count = [sum(count[:i+1]) for i in range(len(count))]
# 将新增数量转换为累积数量
# count[:i+1] 表示从开始到第i个元素（包含）
# 结果：[5661, 12614, 18963, 26099, 32686, 38051, 43004, 48075, 51384, 54585]

# ===== 创建双轴图表 =====
fig, ax2 = plt.subplots()
# 创建图形和第二个坐标轴（用于柱状图）
# fig: 整个图形对象
# ax2: 第二个坐标轴，用于绘制三元组数量

ax1 = ax2.twinx()
# 创建与ax2共享x轴的第一个坐标轴（用于折线图）
# twinx(): 创建一个新的坐标轴，与原坐标轴共享x轴，但有独立的y轴

# ===== 设置第一个坐标轴（准确率）=====
ax1.set_ylim(66,76)
# 设置左侧y轴（准确率）的显示范围：66%到76%

# 绘制DALK方法的折线图
ax1.plot(years,ours, color=color1, marker='o', label='DALK')
# x轴：years, y轴：ours
# color: 橙色, marker: 圆点, label: 图例标签

# 绘制无自感知检索方法的折线图
ax1.plot(years,no_self, color=color2, marker='v', label='w/o self-aware knowledge retrieval')
# x轴：years, y轴：no_self  
# color: 绿色, marker: 倒三角, label: 图例标签

# 绘制基线方法的折线图
ax1.plot(years,baseline, color='blue', linestyle='--', label='Baseline')
# x轴：years, y轴：baseline
# color: 蓝色, linestyle: 虚线, label: 图例标签

# ===== 绘制第二个坐标轴（三元组数量）=====
ax2.bar(years, count,color=color3)
# 绘制柱状图：x轴为年份，y轴为累积三元组数量
# color: 浅蓝色

# ===== 设置坐标轴参数 =====
ax1.tick_params(axis='y', labelsize=16)
# 设置左侧y轴（准确率）的刻度标签字体大小为16

ax2.tick_params(axis='x', labelsize=16)
# 设置x轴（年份）的刻度标签字体大小为16

ax2.tick_params(axis='y', labelsize=16)
# 设置右侧y轴（三元组数量）的刻度标签字体大小为16

# ===== 设置坐标轴标签 =====
ax1.set_xlabel('Year', size=16)
# 设置x轴标签为"Year"，字体大小16

ax1.set_ylabel('Accuracy (%)', size=16)
# 设置左侧y轴标签为"Accuracy (%)"，字体大小16

ax2.set_ylabel('Triplet Number (#)', size=16)
# 设置右侧y轴标签为"Triplet Number (#)"，字体大小16

# ===== 添加图例 =====
ax1.legend(loc='upper left', prop = {'size':16})
# 在左上角添加图例
# loc: 图例位置
# prop: 图例属性，这里设置字体大小为16

# ===== 显示和保存图表 =====
plt.show()
# 显示图表在屏幕上

plt.savefig('evolution.png')
# 将图表保存为PNG文件，文件名为'evolution.png'