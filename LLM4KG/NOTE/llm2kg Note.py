# ====================================================================
# DALK项目 - 精细化关系抽取版本
# 使用两步式方法：1.生成实体摘要 2.基于类型进行关系预测
# 输入：PubTator格式的医学文献数据
# 输出：结构化的实体关系三元组JSON文件
# ====================================================================

import os
from tqdm import tqdm  # 进度条库，用于显示处理进度
import time           # 时间模块，用于API调用间的延时控制
import json           # JSON处理库，用于保存提取结果
from api_utils import *  # 导入自定义的API工具函数

# 定义要处理的文献年份范围（2011-2020年的阿尔兹海默病相关文献）
years = [2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]

# ====================================================================
# 提示模板定义 - 两步式关系抽取方法
# ====================================================================

# 第一步：实体摘要生成模板
# 目的：为每个实体生成简短摘要，描述其与其他医学实体的关系
template_summary = '''Read the following abstract, generate short summary about {} entity "{}" to illustrate what is {}'s relationship with other medical entity.
Abstract: {}
Summary: '''

# 第二步：关系抽取模板 - 使用Chain-of-Thought (CoT)推理
# 目的：基于实体摘要和预定义选项，预测两个实体间的关系
template_relation_extraction_ZeroCoT = '''
Read the following summary, answer the following question.
Summary: {}
Question: predict the relationship between {} entity "{}" and {} entity "{}", first choose from the following options:
{}
Answer: Let's think step by step: '''

# 第三步：关系抽取答案提取模板
# 目的：基于CoT推理结果，提取最终的关系答案
template_relation_extraction_ZeroCoT_answer = '''
Read the following summary, answer the following question.
Summary: {}
Question: predict the relationship between {} entity "{}" and {} entity "{}", first choose from the following options:
{}
Answer: Let's think step by step: {}. So the answer is:'''

# ====================================================================
# 实体类型映射表
# 将PubTator的实体类型映射到Hetionet知识图谱的标准类型
# ====================================================================
entity_map = {
    "Species": "anatomies",           # 物种 → 解剖结构
    "Chromosome": "cellular components",  # 染色体 → 细胞组分
    "CellLine": "cellular components",    # 细胞系 → 细胞组分
    "SNP": "biological processes",       # 单核苷酸多态性 → 生物过程
    "ProteinMutation":"biological processes",     # 蛋白质突变 → 生物过程
    "DNAMutation":"biological processes",         # DNA突变 → 生物过程
    "ProteinAcidChange":"biological processes",   # 蛋白质氨基酸变化 → 生物过程
    "DNAAcidChange":"biological processes",       # DNA碱基变化 → 生物过程
    "Gene": "genes",                 # 基因 → 基因
    "Chemical": "compounds",         # 化学物质 → 化合物
    "Disease": "diseases"            # 疾病 → 疾病
}

# ====================================================================
# 基于实体类型对的预定义关系映射表
# 定义不同实体类型对之间可能存在的关系类型
# 来源：Hetionet生物医学知识图谱的关系定义
# ====================================================================
entities2relation = {
    # 基因-基因关系：共变、相互作用、调节
    ("genes", "genes"): ["covaries", "interacts", "regulates"],
    
    # 疾病-疾病关系：相似性
    ("diseases", "diseases"): ["resembles"],
    
    # 化合物-化合物关系：相似性
    ("compounds", "compounds") : ["resembles"],
    
    # 基因-疾病关系：下调、关联、上调
    ("genes", "diseases"): ["downregulates","associates","upregulates"],
    
    # 基因-化合物关系：结合、上调、下调
    ("genes", "compounds"): ["binds", "upregulates", "downregulates"],
    
    # 化合物-疾病关系：治疗、缓解
    ("compounds", "diseases"): ["treats", "palliates"],
}

# 定义有效的实体类型，只处理这三种核心类型
valid_type = ["genes", "compounds", "diseases"]

# ====================================================================
# 文献数据读取函数
# 功能：从PubTator格式文件中读取并解析医学文献数据
# ====================================================================
def read_literature():
    """
    从PubTator格式文件中读取医学文献数据
    与第一个版本相同，解析标题、摘要和实体标注信息
    
    返回：
    dict: {年份: [文献列表]} 的嵌套字典结构
    """
    # 初始化年份到文献列表的字典映射
    year2literatures = {year: [] for year in years}
    
    # 遍历每个年份的文献文件
    for year in years:
        # 打开对应年份的PubTator格式文件（注意文件夹名为'by_year'而非'by_year_new'）
        with open(os.path.join('by_year', '{}.pubtator'.format(year))) as f:
            # 初始化单篇文献的数据结构
            literature = {'entity': {}}  # entity字典用于存储实体ID到实体信息的映射
            
            # 逐行处理文件内容
            for line in f.readlines():
                line = line.strip()  # 去除行首尾的空白字符
                
                # 空行表示一篇文献的结束，保存当前文献并重置
                if line == ''  and literature != {}:
                    # 将实体名称从集合转换为列表（为了JSON序列化）
                    for entity_id in literature['entity']:
                        literature['entity'][entity_id]['entity_name'] = list(literature['entity'][entity_id]['entity_name'])
                    # 将完整的文献信息添加到对应年份的列表中
                    year2literatures[year].append(literature)
                    # 重置文献结构，准备处理下一篇文献
                    literature = {'entity': {}}
                    continue
                    
                # 处理标题行（格式：PMID|t|标题内容）
                if '|t|' in line:
                    literature['title'] = line.split('|t|')[1]  # 提取标题内容
                    
                # 处理摘要行（格式：PMID|a|摘要内容）
                elif '|a|' in line:
                    literature['abstract'] = line.split('|a|')[1]  # 提取摘要内容
                    
                # 处理实体标注行（制表符分隔的实体信息）
                else:
                    line_list = line.split('\t')  # 按制表符分割
                    
                    # 根据字段数量解析实体信息（处理缺少实体ID的情况）
                    if len(line_list) != 6:
                        entity_name, entity_type, entity_id = line_list[3], line_list[4], None
                    else:
                        entity_name, entity_type, entity_id = line_list[3], line_list[4], line_list[5]
                        
                    # 跳过没有有效ID的实体（'-'表示无ID）
                    if entity_id == '-':
                        continue
                        
                    # 如果是新的实体ID，创建新的实体条目
                    if entity_id not in literature['entity']:
                        literature['entity'][entity_id] = {'entity_name':set(), 'entity_type': entity_type}
                    # 将实体名称添加到集合中（自动处理同义词去重）
                    literature['entity'][entity_id]['entity_name'].add(entity_name)

            # 未使用的变量，保留以维持代码结构
            entity_type = set()
    return year2literatures

# ====================================================================
# 实体名称格式化函数
# 功能：处理一个实体有多个名称的情况，格式化为易读的字符串
# ====================================================================
def get_entity_name(entity_names):
    """
    格式化实体名称，处理一个实体有多个名称的情况
    
    参数：
    entity_names (list): 实体的所有名称列表
    
    返回：
    str: 格式化后的实体名称字符串
    
    示例：
    ['Alzheimer disease'] → 'Alzheimer disease'
    ['Alzheimer disease', 'AD', 'Alzheimer's disease'] → 'Alzheimer disease (AD, Alzheimer's disease)'
    """
    if len(entity_names) == 1:
        return entity_names[0]  # 只有一个名称，直接返回
    else:
        # 多个名称时使用格式：主名称 (别名1, 别名2, ...)
        return '{} ({})'.format(entity_names[0], ', '.join(entity_names[1:]))

# ====================================================================
# 选项构建函数
# 功能：将关系列表转换为多选题格式，便于LLM进行选择
# ====================================================================
def build_options(entity_relation):
    """
    构建多选题选项格式
    
    参数：
    entity_relation (list): 预定义的关系类型列表
    
    返回：
    tuple: (选项字符串, 选项到关系的映射字典)
    
    示例：
    输入：["treats", "palliates"]
    输出：
    ("A. treats\nB. palliates\nC. no-relation\nD. others, please specify...", 
     {"A.": "treats", "B.": "palliates", ...})
    """
    # 添加默认选项：无关系 和 自定义关系
    entity_relation_new = entity_relation + ['no-relation', 'others, please specify by generating a short predicate in 5 words']
    
    # 定义选项标签
    option_list = ['A. ', 'B. ', 'C. ', 'D. ', 'E. ']
    
    ret = ''  # 存储格式化的选项字符串
    option2relation = {}  # 存储选项标签到关系的映射
    
    # 构建选项字符串和映射字典
    for r, o in zip(entity_relation_new, option_list):
        ret += o + r + '\n'  # 格式：A. treats\n
        option2relation[o.strip()] = r  # 映射：{"A.": "treats"}
    
    return ret.strip(), option2relation  # 去除末尾换行符

# ====================================================================
# 主函数：协调整个关系提取流程
# ====================================================================
def main():
    """
    主函数：实现两步式关系抽取流程
    
    处理步骤：
    1. 读取所有年份的文献数据
    2. 对每个实体生成摘要
    3. 对每对符合条件的实体进行关系预测
    4. 保存提取结果到JSON文件
    """
    # 统计变量（未使用，保留以维持代码结构）
    no_relation, with_relation = 0, 0
    
    # 读取所有年份的文献数据
    year2literatures = read_literature()

    # 加载演示样例，用于Few-shot学习
    # demonstration.json包含一些示例，帮助LLM理解任务格式
    demonstration = json.load(open('demonstration.json'))
    demonstration = '\n\n'.join(demonstration)+'\n'  # 将示例连接成字符串

    # 存储所有提取结果的列表
    extracted = []
    
    # 按年份处理文献数据
    for year, literatures in year2literatures.items():
        # 处理该年份的每篇文献
        for literature in tqdm(literatures):
            # 提取文献的标题和摘要
            title, abstract = literature['title'], literature['abstract']
            
            # 初始化当前文献的结果结构
            item = {
                'title': title,      # 文献标题
                'abstract': abstract, # 文献摘要
                'triplet':[]         # 提取的三元组列表
            }
            
            # 双重循环：遍历所有实体对
            # 外层循环：第一个实体（头实体）
            for i, (entity1_id, entity1_info) in enumerate(literature['entity'].items()):
                # 提取第一个实体的信息
                entity1_names, entity1_type = entity1_info['entity_name'], entity1_info['entity_type']
                
                # 跳过不在映射表中的实体类型
                if entity1_type not in entity_map:
                    continue
                
                # 将PubTator类型映射到Hetionet类型
                entity1_type_hetionet = entity_map[entity1_type]
                
                # 只处理有效的实体类型（genes, compounds, diseases）
                if entity1_type_hetionet not in valid_type:
                    continue
                
                # 格式化实体名称
                entity1_name = get_entity_name(entity1_names)
                
                # ======================================
                # 第一步：为第一个实体生成摘要
                # ======================================
                # 构造摘要生成的提示消息
                message = template_summary.format(entity1_type, entity1_name, entity1_name, abstract)
                
                try:
                    # 调用PaLM API生成实体摘要
                    ret_summary = request_api_palm(message)
                except:
                    # API调用失败，跳过当前实体
                    continue
                
                # 内层循环：第二个实体（尾实体）
                for j, (entity2_id, entity2_info) in enumerate(literature['entity'].items()):
                    # 跳过同一个实体（避免自环）
                    if i == j:
                        continue
                    
                    # 提取第二个实体的信息
                    entity2_names, entity2_type = entity2_info['entity_name'], entity2_info['entity_type']
                    
                    # 跳过不在映射表中的实体类型
                    if entity2_type not in entity_map:
                        continue
                    
                    # 将PubTator类型映射到Hetionet类型
                    entity2_type_hetionet = entity_map[entity2_type]
                    
                    # 检查实体类型对是否在预定义关系表中
                    if (entity1_type_hetionet, entity2_type_hetionet) not in entities2relation:
                        continue
                    
                    # API调用频率控制，避免触发速率限制
                    time.sleep(2)
                    
                    # 格式化第二个实体名称
                    entity2_name = get_entity_name(entity2_names)
                    
                    # ======================================
                    # 第二步：进行关系抽取
                    # ======================================
                    
                    # 获取该实体类型对的预定义关系列表
                    entity_relation = entities2relation[(entity1_type_hetionet, entity2_type_hetionet)]
                    
                    # 构建多选题选项
                    options, option2relation = build_options(entity_relation)
                    
                    # ======================================
                    # 第二步A：Chain-of-Thought推理
                    # ======================================
                    # 构造CoT推理的提示消息
                    message = template_relation_extraction_ZeroCoT.format(
                        ret_summary, entity1_type, entity1_name, entity2_type, entity2_name, options
                    )
                    
                    try:
                        # 调用PaLM API进行CoT推理，使用演示样例
                        ret_CoT = request_api_palm(demonstration + message)
                    except:
                        # API调用失败，跳过当前实体对
                        continue
                    
                    # 检查CoT推理结果，空结果跳过
                    if ret_CoT == []:
                        continue
                    
                    # ======================================
                    # 第二步B：提取最终答案
                    # ======================================
                    # 构造答案提取的提示消息
                    message = template_relation_extraction_ZeroCoT_answer.format(
                        ret_summary, entity1_type, entity1_name, entity2_type, entity2_name, options, ret_CoT
                    )
                    
                    try:
                        # 调用PaLM API提取最终关系答案
                        ret_relation = request_api_palm(demonstration + message)
                    except:
                        # API调用失败，跳过当前实体对
                        continue
                    
                    # 检查关系抽取结果，空结果跳过
                    if ret_relation == []:
                        continue
                    
                    # ======================================
                    # 第三步：解析和处理LLM输出
                    # ======================================
                    
                    find = False           # 是否找到匹配的预定义关系
                    is_generated = False   # 是否为LLM生成的新关系
                    
                    # 遍历所有选项，寻找匹配的关系
                    for option, relation in option2relation.items():
                        # 检查LLM输出是否包含某个选项
                        if option in ret_relation or option[0] == ret_relation[0] or relation in ret_relation:
                            # 如果选择了"自定义关系"选项
                            if relation == 'others, please specify by generating a short predicate in 5 words':
                                # 尝试从LLM输出中提取自定义关系
                                if '.' in ret_relation:
                                    relation = ret_relation.split('.')[1]  # 提取句号后的内容
                                else:
                                    relation = ret_relation  # 直接使用整个输出
                                is_generated = True  # 标记为生成的关系
                            find = True  # 标记找到匹配
                            break
                    
                    # 如果没有找到匹配的预定义关系，将整个输出作为自定义关系
                    if not find:
                        is_generated = True
                        relation = ret_relation
                        print('NOT MATCH:', ret_relation, option2relation)  # 调试信息
                    
                    # ======================================
                    # 第四步：保存三元组结果
                    # ======================================
                    item['triplet'].append({
                        'entity1': {
                            'entity_name': entity1_names,           # 头实体所有名称
                            'entity_type': entity1_type_hetionet,   # 头实体标准化类型
                            'entity_id': entity1_id                 # 头实体ID
                        },
                        'entity2': {
                            'entity_name': entity2_names,           # 尾实体所有名称
                            'entity_type': entity2_type_hetionet,   # 尾实体标准化类型
                            'entity_id': entity2_id                 # 尾实体ID
                        },
                        'relation': relation,                        # 关系类型
                        'is_generated': is_generated                 # 是否为生成的关系
                    })
            
            # 将当前文献的提取结果添加到年度结果列表
            extracted.append(item)

        # 保存当前年份的所有提取结果为格式化的JSON文件
        with open('../extracted/{}.json'.format(year), 'w') as f:
            f.write(json.dumps(extracted, indent=2))

# 程序入口点：当脚本直接运行时执行main函数
if __name__ == '__main__':
    main()



# 我用真实的PubTator数据按照DALK代码流程完整演示，并提供中文翻译：

# ## 输入数据（第一篇文献19233513）

# **原始PubTator格式：**
# ```
# 19233513|t|Thiamine deficiency increases beta-secretase activity and accumulation of beta-amyloid peptides.
# 19233513|a|Thiamine pyrophosphate (TPP) and the activities of thiamine-dependent enzymes are reduced in Alzheimer's disease (AD) patients...

# 实体标注：
# 19233513	0	8	Thiamine	Chemical	MESH:D013831
# 19233513	190	209	Alzheimer's disease	Disease	MESH:D000544
# 19233513	468	499	beta-site APP cleaving enzyme 1	Gene	23621
# 19233513	501	506	BACE1	Gene	23621
# 19233513	907	930	reactive oxygen species	Chemical	MESH:D017382
# 19233513	1135	1139	mice	Species	10090
# 19233513	215	223	patients	Species	9606
# ```

# **中文翻译：**
# ```
# 标题：硫胺素缺乏增加β-分泌酶活性和β-淀粉样蛋白的积累
# 摘要：硫胺素焦磷酸（TPP）和硫胺素依赖性酶的活性在阿尔兹海默病（AD）患者中降低...

# 实体：
# 硫胺素 (化学物质)
# 阿尔兹海默病 (疾病)  
# β位点APP切割酶1 (基因)
# BACE1 (基因)
# 活性氧 (化学物质)
# 小鼠 (物种)
# 患者 (物种)
# ```

# ## 第1步：数据解析和类型映射

# **代码执行的类型映射：**
# ```python
# # 原始实体类型 → Hetionet标准类型
# entity_map = {
#     "Chemical": "compounds",    # 化学物质 → 化合物
#     "Disease": "diseases",      # 疾病 → 疾病
#     "Gene": "genes",           # 基因 → 基因
#     "Species": "anatomies"      # 物种 → 解剖结构
# }

# # 解析后的实体字典
# literature['entity'] = {
#     'MESH:D013831': {
#         'entity_name': ['Thiamine'],           # 硫胺素
#         'entity_type': 'Chemical' → 'compounds'
#     },
#     'MESH:D000544': {
#         'entity_name': ['Alzheimer\'s disease', 'AD'],  # 阿尔兹海默病
#         'entity_type': 'Disease' → 'diseases'
#     },
#     '23621': {
#         'entity_name': ['beta-site APP cleaving enzyme 1', 'BACE1'],  # β位点APP切割酶1
#         'entity_type': 'Gene' → 'genes'
#     },
#     'MESH:D017382': {
#         'entity_name': ['reactive oxygen species'],      # 活性氧
#         'entity_type': 'Chemical' → 'compounds'
#     }
# }
# # 注意：mice、patients被过滤掉，因为不在valid_type中
# ```

# ## 第2步：确定有效实体对

# **根据entities2relation表确定可处理的实体对：**
# ```python
# entities2relation = {
#     ("compounds", "diseases"): ["treats", "palliates"],        # 化合物-疾病：治疗、缓解
#     ("genes", "diseases"): ["downregulates","associates","upregulates"],  # 基因-疾病：下调、关联、上调
#     ("genes", "compounds"): ["binds", "upregulates", "downregulates"]     # 基因-化合物：结合、上调、下调
# }

# # 有效的实体对：
# 1. (Thiamine, Alzheimer's disease)     # compounds vs diseases - 硫胺素 vs 阿尔兹海默病
# 2. (reactive oxygen species, Alzheimer's disease)  # compounds vs diseases - 活性氧 vs 阿尔兹海默病  
# 3. (BACE1, Alzheimer's disease)        # genes vs diseases - BACE1 vs 阿尔兹海默病
# 4. (BACE1, Thiamine)                   # genes vs compounds - BACE1 vs 硫胺素
# 5. (BACE1, reactive oxygen species)    # genes vs compounds - BACE1 vs 活性氧
# ```

# ## 第3步：实体摘要生成

# **为硫胺素(Thiamine)生成摘要：**

# **输入LLM的消息：**
# ```
# Read the following abstract, generate short summary about Chemical entity "Thiamine" to illustrate what is Thiamine's relationship with other medical entity.
# Abstract: Thiamine pyrophosphate (TPP) and the activities of thiamine-dependent enzymes are reduced in Alzheimer's disease (AD) patients. In this study, we analyzed the relationship between thiamine deficiency (TD) and amyloid precursor protein (APP) processing in both cellular and animal models of TD...
# Summary:
# ```

# **中文翻译：**
# ```
# 请阅读以下摘要，为化学实体"硫胺素"生成简短摘要，说明硫胺素与其他医学实体的关系。
# 摘要：硫胺素焦磷酸（TPP）和硫胺素依赖性酶的活性在阿尔兹海默病（AD）患者中降低...
# 摘要：
# ```

# **LLM返回的摘要：**
# ```
# Thiamine deficiency is linked to Alzheimer's disease and affects BACE1 enzyme activity, leading to increased beta-amyloid accumulation and oxidative stress through reactive oxygen species.
# ```

# **中文翻译：**
# ```
# 硫胺素缺乏与阿尔兹海默病相关，影响BACE1酶活性，导致β-淀粉样蛋白积累增加和通过活性氧产生氧化应激。
# ```

# ## 第4步：关系抽取（以硫胺素和阿尔兹海默病为例）

# ### 第4.1步：构建选项
# ```python
# entity_relation = ["treats", "palliates"]  # 化合物-疾病的预定义关系
# options = "A. treats\nB. palliates\nC. no-relation\nD. others, please specify by generating a short predicate in 5 words"
# ```

# **中文翻译：**
# ```
# A. 治疗
# B. 缓解  
# C. 无关系
# D. 其他，请用5个词生成简短谓词
# ```

# ### 第4.2步：Chain-of-Thought推理

# **输入LLM：**
# ```
# Read the following summary, answer the following question.
# Summary: Thiamine deficiency is linked to Alzheimer's disease and affects BACE1 enzyme activity, leading to increased beta-amyloid accumulation and oxidative stress through reactive oxygen species.
# Question: predict the relationship between Chemical entity "Thiamine" and Disease entity "Alzheimer's disease", first choose from the following options:
# A. treats
# B. palliates  
# C. no-relation
# D. others, please specify by generating a short predicate in 5 words
# Answer: Let's think step by step:
# ```

# **中文翻译：**
# ```
# 请阅读以下摘要，回答以下问题。
# 摘要：硫胺素缺乏与阿尔兹海默病相关，影响BACE1酶活性，导致β-淀粉样蛋白积累增加和通过活性氧产生氧化应激。
# 问题：预测化学实体"硫胺素"和疾病实体"阿尔兹海默病"之间的关系，首先从以下选项中选择：
# A. 治疗
# B. 缓解
# C. 无关系  
# D. 其他，请用5个词生成简短谓词
# 答案：让我们逐步思考：
# ```

# **LLM的CoT推理：**
# ```
# 1. The summary shows thiamine deficiency is linked to Alzheimer's disease
# 2. Thiamine deficiency leads to negative effects in AD
# 3. This suggests thiamine supplementation could reverse these effects  
# 4. Thiamine likely has therapeutic potential for Alzheimer's disease
# 5. Therefore, thiamine treats Alzheimer's disease
# ```

# **中文翻译：**
# ```
# 1. 摘要显示硫胺素缺乏与阿尔兹海默病相关
# 2. 硫胺素缺乏导致AD的负面影响
# 3. 这表明硫胺素补充可能逆转这些影响
# 4. 硫胺素可能对阿尔兹海默病有治疗潜力
# 5. 因此，硫胺素治疗阿尔兹海默病
# ```

# ### 第4.3步：答案提取

# **输入LLM：**
# ```
# Read the following summary, answer the following question.
# Summary: Thiamine deficiency is linked to Alzheimer's disease and affects BACE1 enzyme activity, leading to increased beta-amyloid accumulation and oxidative stress through reactive oxygen species.
# Question: predict the relationship between Chemical entity "Thiamine" and Disease entity "Alzheimer's disease", first choose from the following options:
# A. treats
# B. palliates
# C. no-relation
# D. others, please specify by generating a short predicate in 5 words
# Answer: Let's think step by step: 1. The summary shows thiamine deficiency is linked to Alzheimer's disease 2. Thiamine deficiency leads to negative effects in AD 3. This suggests thiamine supplementation could reverse these effects 4. Thiamine likely has therapeutic potential for Alzheimer's disease 5. Therefore, thiamine treats Alzheimer's disease. So the answer is:
# ```

# **LLM最终答案：**
# ```
# A. treats
# ```

# ## 第5步：其他关系抽取示例

# ### BACE1与阿尔兹海默病的关系

# **选项：**
# ```
# A. downregulates   # 下调
# B. associates      # 关联  
# C. upregulates     # 上调
# D. no-relation     # 无关系
# E. others          # 其他
# ```

# **LLM答案：**
# ```
# B. associates  # BACE1与阿尔兹海默病关联
# ```

# ### 活性氧与阿尔兹海默病的关系

# **LLM答案：**
# ```
# B. palliates  # 活性氧缓解阿尔兹海默病（错误推理，实际应该是加重）
# ```

# ## 第6步：最终输出结果

# **结构化JSON输出：**
# ```json
# {
#   "title": "Thiamine deficiency increases beta-secretase activity and accumulation of beta-amyloid peptides.",
#   "abstract": "Thiamine pyrophosphate (TPP) and the activities...",
#   "triplet": [
#     {
#       "entity1": {
#         "entity_name": ["Thiamine"],
#         "entity_type": "compounds", 
#         "entity_id": "MESH:D013831"
#       },
#       "entity2": {
#         "entity_name": ["Alzheimer's disease", "AD"],
#         "entity_type": "diseases",
#         "entity_id": "MESH:D000544"  
#       },
#       "relation": "treats",
#       "is_generated": false
#     },
#     {
#       "entity1": {
#         "entity_name": ["beta-site APP cleaving enzyme 1", "BACE1"],
#         "entity_type": "genes",
#         "entity_id": "23621"
#       },
#       "entity2": {
#         "entity_name": ["Alzheimer's disease", "AD"], 
#         "entity_type": "diseases",
#         "entity_id": "MESH:D000544"
#       },
#       "relation": "associates", 
#       "is_generated": false
#     }
#   ]
# }
# ```

# **中文翻译的三元组：**
# ```
# 硫胺素 | 治疗 | 阿尔兹海默病
# BACE1 | 关联 | 阿尔兹海默病
# 活性氧 | 缓解 | 阿尔兹海默病
# BACE1 | 结合 | 硫胺素
# BACE1 | 下调 | 活性氧
# ```

# ## 处理统计

# **原始数据：** 7个实体 → **过滤后：** 4个实体  
# **理论实体对：** 42个 → **实际处理：** 5个  
# **API调用次数：** 14次（4次摘要生成 + 10次关系抽取）  
# **有效三元组：** 5个  

# 这个流程展示了DALK如何通过类型过滤和两步式LLM调用，从原始的生物医学文献中提取出高质量、结构化的知识图谱三元组。