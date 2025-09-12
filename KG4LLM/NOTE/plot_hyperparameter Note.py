# ===== 导入绘图库 =====
from matplotlib import pyplot as plt
# 导入matplotlib的pyplot模块，用于创建图表

plt.style.use('seaborn')
# 设置绘图风格为seaborn，提供更美观的科学图表样式

import seaborn
# 导入seaborn统计绘图库
# seaborn.set(rc={'axes.facecolor':'gainsboro'})
# （注释掉）原本用于设置坐标轴背景为浅灰色

# ===== 定义颜色方案 =====
color1 = "#808080"  # 灰色，用于平均值曲线
color2 = "#038355"  # 绿色，用于MedQA数据集曲线
color3 = '#9BB8F2'  # 蓝色，用于MedMCQA数据集曲线
color4 = "#ec661f"  # 橙色，用于QA4MRE数据集曲线

# ===== 创建图表和坐标轴 =====
fig, ax1 = plt.subplots()
# 创建图形对象和坐标轴
# fig: 整个图形容器
# ax1: 主坐标轴，用于绘制所有曲线

# ===== 设置y轴范围 =====
ax1.set_ylim(50,86)
# 限制y轴显示范围为50%到86%
# 这样可以放大准确率差异，使图表更清晰

# ===== 定义实验数据 =====
k = [1,3,5,10,20,30]
# x轴：k值，表示知识图谱检索中的超参数
# k可能代表：检索的相关知识三元组数量，或者检索路径的跳数

scores_avg = [68.6, 72.0, 72.6, 70.2, 71.6, 71.1]
# 四个数据集的平均准确率
# 对应k值：1→68.6%, 3→72.0%, 5→72.6%, 10→70.2%, 20→71.6%, 30→71.1%

scores_medmc = [ 71.4,71.9,75.2,71.4,72.9,70.4]
# MedMCQA数据集在不同k值下的准确率
# 在k=5时达到最高点75.2%

scores_med = [ 57.9,58.6,57.9,61.8,57.9,57.9]
# MedQA数据集在不同k值下的准确率
# 在k=10时达到最高点61.8%，整体表现相对较低

scores_qa4mre = [ 71.4, 80.0, 71.4, 74.3, 74.3, 74.3]
# QA4MRE数据集在不同k值下的准确率
# 在k=3时达到最高点80.0%，波动较大

# ===== 设置坐标轴刻度参数 =====
ax1.tick_params(axis='y', labelsize=16)
# 设置y轴刻度标签字体大小为16

ax1.tick_params(axis='x', labelsize=16)
# 设置x轴刻度标签字体大小为16

# ===== 绘制各数据集的曲线 =====
ax1.plot(k,scores_med, color=color2, marker='s', label='MedQA')
# 绘制MedQA数据集曲线
# x轴：k值, y轴：scores_med
# color: 绿色, marker: 正方形, label: 图例标签

ax1.plot(k,scores_medmc, color=color3, marker='v', label='MedMCQA')
# 绘制MedMCQA数据集曲线
# x轴：k值, y轴：scores_medmc
# color: 蓝色, marker: 倒三角, label: 图例标签

ax1.plot(k,scores_qa4mre, color=color4, marker='*', label='QA4MRE')
# 绘制QA4MRE数据集曲线
# x轴：k值, y轴：scores_qa4mre
# color: 橙色, marker: 星形, label: 图例标签

ax1.plot(k,scores_avg, color=color1, marker='o', label='AVG')
# 绘制平均值曲线
# x轴：k值, y轴：scores_avg
# color: 灰色, marker: 圆点, label: 图例标签

# ===== 添加图例 =====
ax1.legend(loc='upper right', prop = {'size':16})
# 在右上角添加图例
# loc: 图例位置
# prop: 图例属性字典，设置字体大小为16

# ===== 设置坐标轴标签 =====
ax1.set_xlabel('k', size=20)
# 设置x轴标签为"k"，字体大小20

ax1.set_ylabel('Accuracy (%)', size=20)
# 设置y轴标签为"Accuracy (%)"，字体大小20

# ===== 显示和保存图表 =====
plt.show()
# 在屏幕上显示图表

plt.savefig('hyper-parameter.png')
# 将图表保存为PNG文件，文件名为'hyper-parameter.png'