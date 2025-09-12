# ====================================================================
# DALK项目 - 从DALK生成的知识图谱中提取三元组并转换为标准格式
# 目的：将DALK生成的JSON格式知识图谱转换为TSV格式的三元组文件
# 输入：DALK生成的年度JSON文件 + Hetionet参考数据
# 输出：按年份分别保存的TSV格式三元组文件
# ====================================================================

# 导入必要的库
import json          # 用于处理JSON格式文件
from tqdm import tqdm  # 用于显示处理进度条
import os            # 用于文件路径操作
import pandas as pd  # 用于数据处理和CSV/TSV输出

# ====================================================================
# 第一步：加载Hetionet参考知识图谱数据（用于实体ID到名称的映射）
# ====================================================================

# 读取Hetionet v1.0的完整知识图谱JSON文件
# Hetionet是一个大规模的生物医学异构网络，包含多种实体类型和关系
hetionet = json.load(open('../Hetionet/hetionet-v1.0.json'))

# 构建实体ID到实体名称的映射字典
# 目的：将Hetionet中的实体ID转换为可读的实体名称
id2name = {}

# 遍历Hetionet中的所有节点（实体）
for node in hetionet['nodes']:
    # 每个节点包含：
    # - identifier: 实体的唯一ID
    # - name: 实体的可读名称
    # - kind: 实体类型（如Disease, Gene, Compound等）
    id2name[node['identifier']] = node['name']

# 示例映射：
# 'Disease::DOID:10652' -> 'Alzheimer disease'
# 'Compound::DB00152' -> 'Thiamine'

# ====================================================================
# 第二步：初始化数据处理变量
# ====================================================================

# 全局去重集合，防止跨年份的重复三元组
all_data = set()

# 定义要处理的年份范围（2011-2020年）
years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

# ====================================================================
# 第三步：按年份处理DALK生成的知识图谱数据
# ====================================================================

# 遍历每个年份的数据文件
for year in years:
    # 当前年份的去重集合，用于年度内部去重
    data = set()
    
    # 读取DALK生成式方法产生的该年份知识图谱文件
    # 文件命名格式：{year}_s2s.json（s2s表示sequence-to-sequence生成式方法）
    # 例如：2011_s2s.json, 2012_s2s.json等
    augmented = json.load(open(os.path.join('..', 'extracted', '{}_s2s.json'.format(year))))
    
    # 遍历该年份文件中的每篇文献
    for literature in tqdm(augmented):
        # 每个literature包含：
        # - title: 文献标题
        # - abstract: 文献摘要
        # - triplet: 从该文献提取的三元组列表
        
        # 遍历该文献中提取的每个三元组
        for triplet in literature['triplet']:
            # 每个triplet包含：
            # - entity1: 头实体信息 {'entity_name': 实体名称}
            # - entity2: 尾实体信息 {'entity_name': 实体名称}
            # - relation: 关系类型
            
            # 提取三元组的三个组成部分
            entity1_list = triplet['entity1']['entity_name']  # 头实体名称
            entity2_list = triplet['entity2']['entity_name']  # 尾实体名称
            relation = triplet['relation']                    # 关系类型
            
            # 过滤掉"无关系"的三元组
            # 'no-relation'表示LLM判断两个实体之间没有有意义的关系
            if relation == 'no-relation':
                continue
            
            # 构造三元组元组
            triplet_tuple = (entity1_list, relation, entity2_list)
            
            # 全局去重检查：确保同一个三元组不会在多个年份中重复计入
            if triplet_tuple not in all_data:
                data.add(triplet_tuple)      # 添加到当前年份的数据集
                all_data.add(triplet_tuple)  # 添加到全局数据集，防止后续重复

    # ====================================================================
    # 第四步：将当前年份的三元组数据转换为DataFrame格式
    # ====================================================================
    
    # 将集合数据转换为字典格式，便于创建DataFrame
    data = {
        'head': [item[0] for item in data],     # 头实体列表
        'relation': [item[1] for item in data], # 关系列表
        'tail': [item[2] for item in data],     # 尾实体列表
    }

    # 将字典转换为pandas DataFrame
    # DataFrame提供了方便的数据操作和文件输出功能
    data = pd.DataFrame(data)
    
    # ====================================================================
    # 第五步：将DataFrame保存为TSV（Tab-Separated Values）格式文件
    # ====================================================================
    
    # 保存为TSV文件，使用制表符作为分隔符
    # 参数说明：
    # - header=False: 不输出列名作为第一行
    # - index=False: 不输出行索引
    # - sep='\t': 使用制表符作为字段分隔符
    data.to_csv('extracted_triplet_{}.txt'.format(year), 
                header=False, 
                index=False, 
                sep='\t')
    
    # 输出文件格式示例：
    # Thiamine    treats    Alzheimer's disease
    # BACE1       associates    Alzheimer's disease
    # aspirin     treats    headache

# ====================================================================
# 代码执行结果说明
# ====================================================================
# 
# 执行完成后，会在当前目录生成10个TSV文件：
# - extracted_triplet_2011.txt
# - extracted_triplet_2012.txt
# - ...
# - extracted_triplet_2020.txt
#
# 每个文件包含该年份从文献中提取的所有有效三元组，
# 格式为：头实体\t关系\t尾实体
#
# 这些文件可以用于：
# 1. 知识图谱的进一步分析和可视化
# 2. 与其他知识图谱进行对比评估
# 3. 作为下游机器学习任务的输入数据
# 4. 进行知识图谱质量评估和统计分析

# ====================================================================
# 注意事项
# ====================================================================
# 1. 全局去重机制确保了跨年份的三元组唯一性
# 2. 过滤了'no-relation'类型，只保留有意义的关系
# 3. 使用TSV格式便于后续的数据处理和分析
# 4. Hetionet数据虽然加载了，但在当前代码中实际未使用
#    （可能是为后续的实体标准化功能预留的）