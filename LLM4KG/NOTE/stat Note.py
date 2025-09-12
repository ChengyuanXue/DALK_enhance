# ====================================================================
# DALK项目 - 两种知识图谱构建方法的对比分析脚本
# 目的：统计和对比生成式方法vs配对式方法的KG构建效果
# 输入：两种方法产生的JSON结果文件
# 输出：实体数量、关系类型数量、三元组数量的对比统计
# ====================================================================

# 导入依赖模块
from llm2kg_s2s import read_literature  # 从生成式方法模块导入文献读取函数
import json                              # JSON文件处理库

# ====================================================================
# 第一步：读取原始文献数据，获取基础统计信息
# ====================================================================

# 调用read_literature()函数读取所有年份的PubTator格式文献数据
# 返回格式：{年份: [文献列表]} 的嵌套字典
year2literatures = read_literature()

# 打印每年的文献数量统计
# {k:len(v) for k, v in year2literatures.items()} 
# 创建字典推导式，计算每年文献的数量
print({k:len(v) for k, v in year2literatures.items()})

# 计算所有年份的文献总数
# sum([len(v) for v in year2literatures.values()])
# 对所有年份的文献数量求和
total_literatures = sum([len(v) for v in year2literatures.values()])

# ====================================================================
# 第二步：初始化统计变量 - 用于收集两种方法的KG构建结果
# ====================================================================

# 生成式方法 (Generative Method) 的统计集合
generative_relation = set()  # 存储所有关系类型（去重）
generative_triples = set()   # 存储所有三元组（去重）
generative_node = set()      # 存储所有实体节点（去重）

# 配对式方法 (Pair-wise Method) 的统计集合
pair_relation = set()        # 存储所有关系类型（去重）
pair_triples = set()         # 存储所有三元组（去重）
pair_node = set()            # 存储所有实体节点（去重）

# ====================================================================
# 第三步：遍历所有年份，统计两种方法的KG构建结果
# ====================================================================

# 遍历2011-2020年（共10年）的数据
for year in range(2011, 2021):
    
    # ================================================================
    # 3.1 处理生成式方法的结果文件
    # ================================================================
    
    # 读取生成式方法的JSON结果文件
    # 文件命名格式：{year}_s2s.json （s2s = sequence-to-sequence，生成式）
    # 例如：2011_s2s.json, 2012_s2s.json, ...
    kg_generative = json.load(open('../extracted/{}_s2s.json'.format(year)))
    
    # 遍历该年份的每篇文献的提取结果
    for item_literature in kg_generative:
        # 每个item_literature包含：
        # - title: 文献标题
        # - abstract: 文献摘要
        # - triplet: 提取的三元组列表
        
        # 遍历该文献中提取的每个三元组
        for item in item_literature['triplet']:
            # 每个item（三元组）包含：
            # - entity1: 头实体信息 {'entity_name': 实体名称}
            # - entity2: 尾实体信息 {'entity_name': 实体名称}
            # - relation: 关系类型
            
            # 收集关系类型
            generative_relation.add(item['relation'])
            
            # 收集三元组（头实体名称, 尾实体名称, 关系）
            # 注意：生成式方法中entity_name直接是字符串
            generative_triples.add((
                item['entity1']['entity_name'],    # 头实体名称
                item['entity2']['entity_name'],    # 尾实体名称
                item['relation']                   # 关系类型
            ))
            
            # 收集实体节点
            generative_node.add(item['entity1']['entity_name'])  # 添加头实体
            generative_node.add(item['entity2']['entity_name'])  # 添加尾实体

    # ================================================================
    # 3.2 处理配对式方法的结果文件
    # ================================================================
    
    # 读取配对式方法的JSON结果文件
    # 文件命名格式：{year}_v2.json （v2 = version 2，配对式方法）
    # 例如：2011_v2.json, 2012_v2.json, ...
    kg_pair = json.load(open('../extracted/{}_v2.json'.format(year)))
    
    # 遍历该年份的每篇文献的提取结果
    for item_literature in kg_pair:
        # 数据结构与生成式方法类似，但实体名称格式不同
        
        # 遍历该文献中提取的每个三元组
        for item in item_literature['triplet']:
            # 配对式方法的三元组结构：
            # - entity1: {'entity_name': [名称列表], 'entity_type': 类型, 'entity_id': ID}
            # - entity2: {'entity_name': [名称列表], 'entity_type': 类型, 'entity_id': ID}
            # - relation: 关系类型
            # - is_generated: 是否为生成的关系
            
            # 收集关系类型
            pair_relation.add(item['relation'])
            
            # 收集三元组
            # 注意：配对式方法中entity_name是列表，取第一个元素[0]作为主名称
            pair_triples.add((
                item['entity1']['entity_name'][0],  # 头实体主名称
                item['entity2']['entity_name'][0],  # 尾实体主名称
                item['relation']                    # 关系类型
            ))
            
            # 收集实体节点 - 需要遍历实体名称列表，因为一个实体可能有多个别名
            # 处理头实体的所有名称
            for entity in item['entity1']['entity_name']:
                pair_node.add(entity)  # 添加头实体的每个名称
            
            # 处理尾实体的所有名称  
            for entity in item['entity2']['entity_name']:
                pair_node.add(entity)  # 添加尾实体的每个名称

# ====================================================================
# 第四步：输出统计结果对比
# ====================================================================

# 输出总文献数量
print(total_literatures)

# 输出生成式方法的统计结果
print('generative')  # 生成式方法标题
print('entity:', len(generative_node))      # 实体节点总数
print('relation:', len(generative_relation)) # 关系类型总数  
print('triples:', len(generative_triples))   # 三元组总数

# 输出配对式方法的统计结果
print('pair-wised')  # 配对式方法标题
print('entity:', len(pair_node))       # 实体节点总数
print('relation:', len(pair_relation)) # 关系类型总数
print('triples:', len(pair_triples))   # 三元组总数

# ====================================================================
# 代码执行后的预期输出示例：
# ====================================================================
# {2011: 1234, 2012: 1456, 2013: 1345, ...}  # 每年文献数量
# 12450  # 总文献数量
# generative
# entity: 25847    # 生成式方法提取的实体数量
# relation: 156    # 生成式方法发现的关系类型数量
# triples: 89456   # 生成式方法构建的三元组数量
# pair-wised  
# entity: 18923    # 配对式方法提取的实体数量
# relation: 24     # 配对式方法发现的关系类型数量（较少，因为有预定义约束）
# triples: 67234   # 配对式方法构建的三元组数量

# ====================================================================
# 关键差异说明：
# ====================================================================
# 1. 文件命名差异：
#    - 生成式方法：{year}_s2s.json
#    - 配对式方法：{year}_v2.json
#
# 2. 数据结构差异：
#    - 生成式方法：entity_name 是字符串
#    - 配对式方法：entity_name 是列表，包含多个别名
#
# 3. 统计方式差异：
#    - 生成式方法：直接计数实体名称
#    - 配对式方法：需要展开实体名称列表进行计数
#
# 4. 预期结果差异：
#    - 生成式方法：关系类型更多样化（开放式生成）
#    - 配对式方法：关系类型更规范化（基于预定义类型）
#    - 生成式方法：可能产生更多三元组
#    - 配对式方法：三元组质量更高但数量可能较少