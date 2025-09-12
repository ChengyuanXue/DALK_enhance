# ====================================================================
# DALK项目 - 从生物医学文献中提取知识图谱三元组
# 使用Google PaLM API进行生成式关系提取
# 输入：PubTator格式的医学文献数据
# 输出：结构化的实体关系三元组JSON文件
# ====================================================================

import os
from tqdm import tqdm  # 进度条库，用于显示处理进度
import time           # 时间模块，用于API调用间的延时控制
import json           # JSON处理库，用于保存提取结果
from api_utils import *  # 导入自定义的API工具函数，包含request_api_palm函数

# 定义要处理的文献年份范围（2011-2020年的阿尔兹海默病相关文献）
years = [2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]

# 关系提取的提示模板 - 这是发送给PaLM API的指令
# 模板包含：1.任务描述 2.预定义关系类型 3.输出格式要求 4.示例
template = '''Read the following abstract, extract the relationships between each entity.
You can choose the relation from: (covaries, interacts, regulates, resembles, downregulates, upregulates, associates, binds, treats, palliates), or generate a new predicate to describe the relationship between the two entities.
Output all the extract triples in the format of "head | relation | tail". For example: "Alzheimer's disease | associates | memory deficits"

Abstract: {}
Entity: {}
Output: '''

# template = '''请阅读以下摘要，提取各实体之间的关系。
# 你可以从以下关系中选择：(共变, 相互作用, 调节, 相似, 下调, 上调, 关联, 结合, 治疗, 缓解)，或者生成一个新的谓词来描述两个实体之间的关系。
# 请以"头实体 | 关系 | 尾实体"的格式输出所有提取的三元组。例如："阿尔兹海默病 | 关联 | 记忆缺陷"

# 摘要：{}
# 实体：{}
# 输出：'''

def read_literature():
    """
    从PubTator格式文件中读取医学文献数据
    
    PubTator格式说明：
    - 标题行：PMID|t|标题内容
    - 摘要行：PMID|a|摘要内容  
    - 实体行：PMID\t起始位置\t结束位置\t实体名\t实体类型\t实体ID
    - 空行分隔不同文献
    
    返回：
    dict: {年份: [文献列表]} 的嵌套字典结构
    """
    # 初始化年份到文献列表的字典映射
    year2literatures = {year: [] for year in years}
    
    # 遍历每个年份的文献文件
    for year in years:
        # 打开对应年份的PubTator格式文件
        with open(os.path.join('by_year_new', '{}.pubtator'.format(year))) as f:
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

            # 这行代码在原始代码中存在但未使用，保留以维持原始结构
            entity_type = set()
    return year2literatures

def get_entity_name(entity_names):
    """
    格式化实体名称，处理一个实体有多个名称的情况
    
    参数：
    entity_names (list): 实体的所有名称列表
    
    返回：
    str: 格式化后的实体名称字符串
    """
    if len(entity_names) == 1:
        return entity_names[0]  # 只有一个名称，直接返回
    else:
        # 多个名称时使用格式：主名称 (别名1, 别名2, ...)
        return '{} ({})'.format(entity_names[0], ', '.join(entity_names[1:]))

def main():
    """
    主函数：协调整个关系提取流程
    
    处理步骤：
    1. 读取所有年份的文献数据
    2. 对每篇文献进行关系提取
    3. 保存提取结果到JSON文件
    """
    # 统计变量（原始代码中定义但未使用）
    no_relation, with_relation = 0, 0
    
    # 读取所有年份的文献数据
    year2literatures = read_literature()

    # 按年份处理文献数据
    for year, literatures in tqdm(year2literatures.items()):
        extracted = []  # 存储当前年份提取的所有结果
        
        # 处理该年份的每篇文献
        for literature in tqdm(literatures):
            # API调用频率控制，避免触发速率限制
            time.sleep(1)
            
            # 提取文献的标题和摘要
            title, abstract = literature['title'], literature['abstract']
            
            # 初始化当前文献的结果结构
            item = {
                'title': title,      # 文献标题
                'abstract': abstract, # 文献摘要
                'triplet':[]         # 提取的三元组列表
            }
            
            # 构造实体列表字符串，将所有实体名称连接成逗号分隔的字符串
            entity_names = ', '.join([get_entity_name(entity_info['entity_name']) for entity_info in literature['entity'].values()])
            
            # 构造发送给PaLM API的完整提示消息，将摘要和实体列表填入模板
            message = template.format(abstract, entity_names)
            
            # 调用PaLM API进行关系提取
            try:
                ret = request_api_palm(message)  # 发送请求到PaLM API
            except Exception as E:
                # API调用失败，跳过当前文献
                continue
                
            # 检查API返回结果，空结果跳过
            if ret == []:
                continue
                
            # 解析API返回的三元组字符串
            for triple in ret.split('\n'):  # 按行分割多个三元组
                if triple == '':
                    continue  # 跳过空行
                try:
                    # 按照 " | " 分隔符解析三元组，期望格式：实体1 | 关系 | 实体2
                    entity1, relation, entity2 = triple.split(' | ')
                except:
                    # 解析失败，跳过格式不正确的三元组
                    continue
                    
                # 将解析成功的三元组添加到结果中
                item['triplet'].append({
                    'entity1': {
                        'entity_name': entity1,  # 头实体名称
                    },
                    'entity2': {
                        'entity_name': entity2,  # 尾实体名称
                    },
                    'relation': relation,        # 关系类型
                })
            # 将当前文献的提取结果添加到年度结果列表
            extracted.append(item)

        # 保存当前年份的所有提取结果为格式化的JSON文件
        with open('../extracted/{}_s2s.json'.format(year), 'w') as f:
            f.write(json.dumps(extracted, indent=2))

# 程序入口点：当脚本直接运行时执行main函数
if __name__ == '__main__':
    main()

# 读取的文件在by_year里面，格式如下

# PMID    起始位置    结束位置    实体名称    实体类型    实体ID
# 25847293    45    63    Alzheimer's disease    Disease    MESH:D000544
# 25847293    78    92    memory deficits    Phenotype    HP:0002354
# 25847293    105    119    tau protein    Protein    UniProt:P10636

# 这个文档中有第三方（PubTator）根据文献提取的文献中的实体，并且有全球统一的实体ID
# 但是没有实体之间的关系，这个代码是通过LLM补充实体之间的关系（可以查看提示词理解）