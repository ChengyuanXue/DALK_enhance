# DALK项目核心实现文件 - MindMap知识图谱增强问答系统
# 功能：结合知识图谱和大语言模型进行医学问答

# ==================== 导入所需库 ====================

# LangChain相关库 - 用于LLM调用和提示管理
from langchain.chat_models import ChatOpenAI  # OpenAI聊天模型接口
from langchain import PromptTemplate,LLMChain  # 提示模板和LLM链
from langchain.prompts.chat import (
    ChatPromptTemplate,              # 聊天提示模板
    SystemMessagePromptTemplate,     # 系统消息模板
    AIMessagePromptTemplate,         # AI消息模板
    HumanMessagePromptTemplate,      # 人类消息模板
)
from langchain.schema import (
    AIMessage,      # AI消息类型
    HumanMessage,   # 人类消息类型
    SystemMessage,  # 系统消息类型
)

# 数值计算和数据处理库
import numpy as np          # 数值计算库，用于向量操作
import re                   # 正则表达式库，用于文本解析
import string               # 字符串处理工具
import pandas as pd         # 数据框架，用于数据处理
from collections import deque  # 双端队列数据结构
import itertools            # 迭代工具，用于组合生成
from typing import Dict, List  # 类型提示
import pickle               # 序列化库，用于加载预训练向量
import json                 # JSON处理库
import csv                  # CSV文件处理
import os                   # 操作系统接口
import sys                  # 系统相关功能
from time import sleep      # 时间延迟函数

# Neo4j图数据库相关
from neo4j import GraphDatabase, basic_auth  # Neo4j数据库驱动

# 机器学习和相似度计算
from sklearn.metrics.pairwise import cosine_similarity  # 余弦相似度计算
from sklearn.preprocessing import normalize               # 数据标准化

# OpenAI API
import openai  # OpenAI API接口

# 文本评估指标
from pycocoevalcap.bleu.bleu import Bleu      # BLEU评分
from pycocoevalcap.cider.cider import Cider   # CIDEr评分
from pycocoevalcap.rouge.rouge import Rouge   # ROUGE评分
from pycocoevalcap.meteor.meteor import Meteor # METEOR评分

# LangChain OpenAI接口
from langchain.llms import OpenAI

# 图像处理（用于可视化）
from PIL import Image, ImageDraw, ImageFont

# 文本相似度和检索
from gensim import corpora                           # 语料库处理
from gensim.models import TfidfModel                 # TF-IDF模型
from gensim.similarities import SparseMatrixSimilarity  # 稀疏矩阵相似度
from rank_bm25 import BM25Okapi                      # BM25检索算法
from gensim.models import Word2Vec                   # Word2Vec词向量模型

# 自定义数据集处理工具
from dataset_utils import *  # 导入之前分析的数据集处理器
from tqdm import tqdm        # 进度条显示

# ==================== 全局配置和映射 ====================

# 数据集名称到处理器类的映射字典
dataset2processor = {
    'medmcqa': medmcqaZeroshotsProcessor,  # MedMCQA数据集处理器
    'medqa': medqaZeroshotsProcessor,      # MedQA数据集处理器
    'mmlu': mmluZeroshotsProcessor,        # MMLU数据集处理器
    'qa4mre': qa4mreZeroshotsProcessor     # QA4MRE数据集处理器
}

# 要处理的数据集列表
datasets = ['medqa', 'medmcqa', 'mmlu', 'qa4mre']

# ==================== LLM调用函数 ====================

def chat_35(prompt):
    """
    调用GPT-3.5-turbo模型生成回复
    
    参数:
        prompt: 输入提示词
    
    返回:
        模型生成的文本回复
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",    # 使用GPT-3.5-turbo模型
        messages=[
            {"role": "user", "content": prompt}  # 用户角色发送提示词
        ]
    )
    return completion.choices[0].message.content  # 返回生成的内容

def chat_4(prompt):
    """
    调用GPT-4模型生成回复
    
    参数:
        prompt: 输入提示词
    
    返回:
        模型生成的文本回复
    """
    completion = openai.ChatCompletion.create(
        model="gpt-4",           # 使用GPT-4模型
        messages=[
            {"role": "user", "content": prompt}  # 用户角色发送提示词
        ]
    )
    return completion.choices[0].message.content  # 返回生成的内容

# ==================== 实体提取函数 ====================

def prompt_extract_keyword(input_text):
    """
    使用LLM从医学文本中提取关键实体
    
    参数:
        input_text: 输入的医学文本
    
    返回:
        提取的实体列表
    """
    # 定义few-shot提示模板，包含示例，这个few-shot提示模板的作用是教会LLM从医学对话中识别和提取关键实体
    template = """
    There are some samples:
    \n\n
    ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
    <CLS>Doctor, I have been having discomfort and dryness in my vagina for a while now. I also experience pain during sex. What could be the problem and what tests do I need?<SEP>The extracted entities are\n\n ### Output:
    <CLS>Doctor, I have been having discomfort and dryness in my vagina for a while now. I also experience pain during sex. What could be the problem and what tests do I need?<SEP>The extracted entities are Vaginal pain, Vaginal dryness, Pain during intercourse<EOS>
    \n\n
    Instruction:\n'Learn to extract entities from the following medical answers.'\n\n### Input:\n
    <CLS>Okay, based on your symptoms, we need to perform some diagnostic procedures to confirm the diagnosis. We may need to do a CAT scan of your head and an Influenzavirus antibody assay to rule out any other conditions. Additionally, we may need to evaluate you further and consider other respiratory therapy or physical therapy exercises to help you feel better.<SEP>The extracted entities are\n\n ### Output:
    <CLS>Okay, based on your symptoms, we need to perform some diagnostic procedures to confirm the diagnosis. We may need to do a CAT scan of your head and an Influenzavirus antibody assay to rule out any other conditions. Additionally, we may need to evaluate you further and consider other respiratory therapy or physical therapy exercises to help you feel better.<SEP>The extracted entities are CAT scan of head (Head ct), Influenzavirus antibody assay, Physical therapy exercises; manipulation; and other procedures, Other respiratory therapy<EOS>
    \n\n
    Try to output:
    ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
    <CLS>{input}<SEP>The extracted entities are\n\n ### Output:
    """

    # 创建提示模板对象
    prompt = PromptTemplate(
        template = template,
        input_variables = ["input"]  # 定义输入变量
    )

    # 创建系统消息提示模板
    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(input = input_text)  # 格式化输入文本

    # 创建人类消息模板
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # 组合聊天提示
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(input = input_text,\
                                                        text={})

    # 调用LLM生成回复
    response_of_KG = chat(chat_prompt_with_values.to_messages()).content

    # 使用正则表达式提取实体
    question_kg = re.findall(re1,response_of_KG)
    return question_kg

# ==================== 知识图谱路径查找函数 ====================

def find_shortest_path(start_entity_name, end_entity_name, candidate_list):
    """
    在Neo4j知识图谱中查找两个实体间的最短路径
    
    参数:
        start_entity_name: 起始实体名称
        end_entity_name: 目标实体名称
        candidate_list: 候选实体列表
    
    返回:
        paths: 找到的路径列表
        exist_entity: 路径中存在的候选实体
    """
    global exist_entity  # 使用全局变量存储存在的实体
    
    # 使用Neo4j数据库会话
    with driver.session() as session:
        # Cypher查询：查找两实体间的所有最短路径（最多5跳）
        result = session.run(
            "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
            "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
            "RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        
        paths = []        # 存储路径字符串
        short_path = 0    # 标记是否找到包含候选实体的路径
        
        # 遍历查询结果中的每条路径
        for record in result:
            path = record["p"]    # 获取路径对象
            entities = []         # 存储路径中的实体
            relations = []        # 存储路径中的关系
            
            # 提取路径中的节点（实体）
            for i in range(len(path.nodes)):
                node = path.nodes[i]
                entity_name = node["name"]
                entities.append(entity_name)
                
                # 提取路径中的关系
                if i < len(path.relationships):
                    relationship = path.relationships[i]
                    relation_type = relationship.type
                    relations.append(relation_type)
           
            # 构建路径字符串
            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i].replace("_"," ")  # 将下划线替换为空格
                
                # 检查当前实体是否在候选列表中
                if entities[i] in candidate_list:
                    short_path = 1                    # 标记找到候选实体
                    exist_entity = entities[i]       # 记录存在的实体
                    
                path_str += entities[i]  # 添加实体到路径字符串
                
                # 添加关系到路径字符串
                if i < len(relations):
                    relations[i] = relations[i].replace("_"," ")  # 将下划线替换为空格
                    path_str += "->" + relations[i] + "->"
            
            # 如果找到包含候选实体的路径，优先返回
            if short_path == 1:
                paths = [path_str]
                break
            else:
                paths.append(path_str)
                exist_entity = {}  # 重置存在实体
            
        # 限制返回的路径数量（最多5条）
        if len(paths) > 5:        
            paths = sorted(paths, key=len)[:5]  # 按长度排序，取前5条
        
        try:
            return paths, exist_entity
        except:
            return paths, {}

# ==================== 组合生成函数 ====================

def combine_lists(*lists):
    """
    生成多个列表的笛卡尔积组合
    
    参数:
        *lists: 可变数量的列表参数
    
    返回:
        results: 所有可能的组合列表
    """
    # 生成所有列表的笛卡尔积
    combinations = list(itertools.product(*lists))
    results = []
    
    # 处理每个组合
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            # 如果元素是列表，则扩展到新组合中
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                # 如果是单个元素，直接添加
                new_combination.append(sublist)
        results.append(new_combination)
    return results

# ==================== 实体邻居查找函数 ====================

def get_entity_neighbors(entity_name: str, disease_flag) -> List[List[str]]:
    """
    获取指定实体在知识图谱中的邻居实体
    
    参数:
        entity_name: 实体名称
        disease_flag: 疾病标志，用于过滤某些关系类型
    
    返回:
        neighbor_list: 邻居实体列表
        disease: 疾病相关实体列表
    """
    disease = []  # 存储疾病相关实体
    
    # Cypher查询：获取实体的所有出边关系和邻居
    query = """
    MATCH (e:Entity)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type,
           collect(n.name) AS neighbor_entities
    """
    result = session.run(query, entity_name=entity_name)

    neighbor_list = []  # 存储邻居信息
    
    # 处理查询结果
    for record in result:
        rel_type = record["relationship_type"]     # 关系类型
        
        # 如果是疾病标志且关系类型是症状，则跳过
        if disease_flag == 1 and rel_type == 'has_symptom':
            continue

        neighbors = record["neighbor_entities"]    # 邻居实体列表
        
        # 如果关系类型包含"disease"，添加到疾病列表
        if "disease" in rel_type.replace("_"," "):
            disease.extend(neighbors)
        else:
            # 格式化邻居信息：[实体名, 关系类型, 邻居实体列表]
            neighbor_list.append([entity_name.replace("_"," "), rel_type.replace("_"," "), 
                                ','.join([x.replace("_"," ") for x in neighbors])
                                ])
    
    return neighbor_list, disease

# ==================== 路径信息转换函数 ====================

def prompt_path_finding(path_input):
    """
    将知识图谱路径转换为自然语言描述
    
    参数:
        path_input: 图谱路径信息
    
    返回:
        自然语言描述的路径信息
    """
    # 定义提示模板
    template = """
    There are some knowledge graph path. They follow entity->relationship->entity format.
    \n\n
    {Path}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Path-based Evidence 1, Path-based Evidence 2,...\n\n

    Output:
    """

    # 创建提示模板
    prompt = PromptTemplate(
        template = template,
        input_variables = ["Path"]
    )

    # 创建系统消息提示
    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(Path = path_input)

    # 创建人类消息模板
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # 组合聊天提示
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(Path = path_input,\
                                                        text={})

    # 调用LLM生成自然语言描述
    response_of_KG_path = chat(chat_prompt_with_values.to_messages()).content
    return response_of_KG_path

# ==================== 邻居信息转换函数 ====================

def prompt_neighbor(neighbor):
    """
    将知识图谱邻居信息转换为自然语言描述
    
    参数:
        neighbor: 邻居实体信息
    
    返回:
        自然语言描述的邻居信息
    """
    # 定义提示模板
    template = """
    There are some knowledge graph. They follow entity->relationship->entity list format.
    \n\n
    {neighbor}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2,...\n\n

    Output:
    """

    # 创建提示模板
    prompt = PromptTemplate(
        template = template,
        input_variables = ["neighbor"]
    )

    # 创建系统消息提示
    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(neighbor = neighbor)

    # 创建人类消息模板
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # 组合聊天提示
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(neighbor = neighbor,\
                                                        text={})

    # 调用LLM生成自然语言描述
    response_of_KG_neighbor = chat(chat_prompt_with_values.to_messages()).content

    return response_of_KG_neighbor

# ==================== 知识过滤函数 ====================

def self_knowledge_retrieval(graph, question):
    """
    使用LLM过滤与问题无关的知识图谱信息
    
    参数:
        graph: 知识图谱信息
        question: 问题文本
    
    返回:
        过滤后的知识图谱信息
    """
    # 定义知识过滤提示模板
    template = """
    There is a question and some knowledge graph. The knowledge graphs follow entity->relationship->entity list format.
    \n\n
    ##Graph: {graph}
    \n\n
    ##Question: {question}
    \n\n
    Please filter noisy knowledge from this knowledge graph that useless or irrelevant to the give question. Output the filtered knowledges in the same format as the input knowledge graph.\n\n

    Filtered Knowledge:
    """

    # 创建提示模板
    prompt = PromptTemplate(
        template = template,
        input_variables = ["graph", "question"]
    )

    # 创建系统消息提示
    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(graph = graph, question=question)

    # 创建人类消息模板
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # 组合聊天提示
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(graph = graph, question=question,\
                                                        text={})

    # 调用LLM进行知识过滤
    response_of_KG_neighbor = chat(chat_prompt_with_values.to_messages()).content

    return response_of_KG_neighbor

# ==================== 知识重排序函数 ====================

def self_knowledge_retrieval_reranking(graph, question):
    """
    使用LLM对知识图谱信息进行重排序，选出最相关的三元组
    
    参数:
        graph: 知识图谱信息
        question: 问题文本
    
    返回:
        重排序后的知识图谱信息
    """
    # 定义知识重排序提示模板
    template = """
    There is a question and some knowledge graph. The knowledge graphs follow entity->relationship->entity list format.
    \n\n
    ##Graph: {graph}
    \n\n
    ##Question: {question}
    \n\n
    Please rerank the knowledge graph and output at most 5 important and relevant triples for solving the given question. Output the reranked knowledge in the following format:
    Reranked Triple1: xxx ——> xxx
    Reranked Triple2: xxx ——> xxx
    Reranked Triple3: xxx ——> xxx

    Answer:
    """

    # 创建提示模板
    prompt = PromptTemplate(
        template = template,
        input_variables = ["graph", "question"]
    )

    # 创建系统消息提示
    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(graph = graph, question=question)

    # 创建人类消息模板
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # 组合聊天提示
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(graph = graph, question=question,\
                                                        text={})

    # 调用LLM进行知识重排序
    response_of_KG_neighbor = chat(chat_prompt_with_values.to_messages()).content

    return response_of_KG_neighbor

# ==================== 余弦相似度计算函数 ====================

def cosine_similarity_manual(x, y):
    """
    手动计算两个向量间的余弦相似度
    
    参数:
        x: 第一个向量
        y: 第二个向量
    
    返回:
        相似度矩阵
    """
    # 计算向量点积
    dot_product = np.dot(x, y.T)
    # 计算向量范数
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    # 计算余弦相似度
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim

# ==================== 回答质量检查函数 ====================

def is_unable_to_answer(response):
    """
    检查LLM回复是否无法回答问题
    
    参数:
        response: LLM的回复
    
    返回:
        True: 无法回答，False: 可以回答
    """
    # 使用GPT-3.5评估回复质量
    analysis = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=response,
        max_tokens=1,
        temperature=0.0,
        n=1,
        stop=None,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    
    # 清理评分结果
    score = analysis.choices[0].text.strip().replace("'", "").replace(".", "")
    
    # 如果评分不是数字，认为无法回答
    if not score.isdigit():   
        return True
        
    # 设置质量阈值
    threshold = 0.6
    if float(score) > threshold:
        return False  # 质量较好，可以回答
    else:
        return True   # 质量较差，无法回答

# ==================== 文本自动换行函数 ====================

def autowrap_text(text, font, max_width):
    """
    根据指定宽度自动换行文本（用于图像生成）
    
    参数:
        text: 要换行的文本
        font: 字体对象
        max_width: 最大宽度
    
    返回:
        换行后的文本行列表
    """
    text_lines = []
    
    # 如果文本宽度小于最大宽度，直接返回
    if font.getsize(text)[0] <= max_width:
        text_lines.append(text)
    else:
        # 按单词分割文本
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            # 在不超过最大宽度的情况下添加单词
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + ' '
                i += 1
            # 如果行为空，说明单个单词太长，强制添加
            if not line:
                line = words[i]
                i += 1
            text_lines.append(line)
    return text_lines

# ==================== 最终答案生成函数 ====================

def final_answer(str, response_of_KG_list_path, response_of_KG_neighbor):
    """
    结合知识图谱信息生成最终答案
    
    参数:
        str: 输入问题
        response_of_KG_list_path: 路径信息
        response_of_KG_neighbor: 邻居信息
    
    返回:
        最终生成的答案
    """
    # 处理空的知识图谱信息
    if response_of_KG_list_path == []:
        response_of_KG_list_path = ''
    if response_of_KG_neighbor == []:
        response_of_KG_neighbor = ''
    
    # 第一阶段：生成CoT推理过程
    messages = [
        SystemMessage(content="You are an excellent AI assistant to answering the following question"),
        HumanMessage(content='Question: '+input_text[0]),
        AIMessage(content="You have some medical knowledge information in the following:\n\n" +  '###'+ response_of_KG_list_path + '\n\n' + '###' + response_of_KG_neighbor),
        HumanMessage(content="Answer: Let's think step by step: ")
    ]
    result_CoT = chat(messages)    # 调用LLM生成推理过程
    output_CoT = result_CoT.content
    
    # 第二阶段：基于推理过程生成最终答案
    messages = [
        SystemMessage(content="You are an excellent AI assistant to answering the following question"),
        HumanMessage(content='Question: '+input_text[0]),
        AIMessage(content="You have some medical knowledge information in the following:\n\n" +  '###'+ response_of_KG_list_path + '\n\n' + '###' + response_of_KG_neighbor),
        AIMessage(content="Answer: Let's think step by step: "+output_CoT),
        AIMessage(content="The final answer (output the letter option) is:")
    ]
    result = chat(messages)        # 调用LLM生成最终答案
    output_all = result.content
    return output_all

# ==================== 文档检索函数 ====================

def prompt_document(question, instruction):
    """
    基于问题和医学知识生成诊断建议
    
    参数:
        question: 患者描述
        instruction: 医学知识指导
    
    返回:
        诊断和治疗建议
    """
    # 定义医学诊断提示模板
    template = """
    You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation.\n\n
    Patient input:\n
    {question}
    \n\n
    You have some medical knowledge information in the following:
    {instruction}
    \n\n
    What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease?
    """

    # 创建提示模板
    prompt = PromptTemplate(
        template = template,
        input_variables = ["question","instruction"]
    )

    # 创建系统消息提示
    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(question = question,
                                 instruction = instruction)

    # 创建人类消息模板
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # 组合聊天提示
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(question = question,\
                                                        instruction = instruction,\
                                                        text={})

    # 调用LLM生成诊断建议
    response_document_bm25 = chat(chat_prompt_with_values.to_messages()).content

    return response_document_bm25

# ==================== 主执行程序 ====================

if __name__ == "__main__":
    # ==================== 配置初始化 ====================
    
    # OpenAI API密钥配置
    YOUR_OPENAI_KEY = ''#replace this to your key
    os.environ['OPENAI_API_KEY']= YOUR_OPENAI_KEY  # 设置环境变量
    openai.api_key = YOUR_OPENAI_KEY               # 设置OpenAI API密钥

    # ==================== Neo4j数据库连接配置 ====================
    
    # Neo4j数据库连接参数
    uri = ""#replace this to your neo4j uri           # Neo4j数据库URI
    username = ""#replace this to your neo4j username # 用户名
    password = ""#replace this to your neo4j password # 密码

    # 建立数据库连接
    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()

    # ==================== 构建知识图谱 ====================

    # 清空数据库中的所有节点和关系
    session.run("MATCH (n) DETACH DELETE n")# clean all

    # 读取三元组数据文件
    df = pd.read_csv('./Alzheimers/train_s2s.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])

    # 遍历每个三元组，构建知识图谱
    for index, row in tqdm(df.iterrows()):
        head_name = row['head']      # 头实体
        tail_name = row['tail']      # 尾实体
        relation_name = row['relation']  # 关系名称

        # 构建Cypher查询，创建实体和关系
        query = (
            "MERGE (h:Entity { name: $head_name }) "      # 创建或匹配头实体
            "MERGE (t:Entity { name: $tail_name }) "      # 创建或匹配尾实体
            "MERGE (h)-[r:`" + relation_name + "`]->(t)"  # 创建关系
        )
        try:
            # 执行查询
            session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)
        except:
            continue  # 如果出错，跳过当前三元组

    # ==================== LLM和向量加载配置 ====================

    # 配置LangChain OpenAI接口
    OPENAI_API_KEY = YOUR_OPENAI_KEY
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo', temperature=0.7)

    # 定义正则表达式模式，用于提取实体
    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"

    # 加载预训练的实体向量嵌入
    with open('./Alzheimers/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)
    
    # 加载预训练的关键词向量嵌入    
    with open('./Alzheimers/keyword_embeddings.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)

    # ==================== 主处理循环 ====================

    # 遍历每个数据集进行处理
    for dataset in datasets:
        # 获取对应的数据集处理器
        processor = dataset2processor[dataset]()
        data = processor.load_dataset()  # 加载数据集

        acc, total_num = 0, 0  # 初始化准确率统计
        generated_data=[]      # 存储处理结果

        # 遍历数据集中的每个问题
        for item in tqdm(data):
            # 生成问答提示
            input_text = [processor.generate_prompt(item)]
            
            # 提取实体列表
            entity_list = item['entity'].split('\n')
            question_kg = []
            for entity in entity_list:
                try:
                    # 解析实体（去除序号）
                    entity = entity.split('.')[1].strip()
                    question_kg.append(entity)
                except:
                    continue

            # ==================== 实体匹配 ====================
            
            match_kg = []  # 存储匹配的知识图谱实体
            entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
            
            # 对每个提取的实体进行向量匹配
            for kg_entity in question_kg:
                # 获取实体在关键词嵌入中的索引
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
                kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

                # 计算与所有知识图谱实体的余弦相似度
                cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
                max_index = cos_similarities.argmax()  # 找到最相似的实体
                
                # 获取匹配的实体名称
                match_kg_i = entity_embeddings["entities"][max_index]
                
                # 避免重复匹配同一个实体
                while match_kg_i in match_kg:
                    cos_similarities[max_index] = 0          # 将已匹配实体的相似度设为0
                    max_index = cos_similarities.argmax()   # 重新找最相似的
                    match_kg_i = entity_embeddings["entities"][max_index]
                match_kg.append(match_kg_i)

            # ==================== 知识图谱路径查找 ====================
            
            # 如果匹配到多个实体，进行路径查找
            if len(match_kg) != 1 or 0:
                start_entity = match_kg[0]      # 起始实体
                candidate_entity = match_kg[1:] # 候选实体列表
                
                result_path_list = []  # 存储所有找到的路径
                
                # 路径查找主循环
                while 1:
                    flag = 0           # 标记是否需要重新开始
                    paths_list = []    # 当前轮次的路径列表
                    
                    # 遍历所有候选实体
                    while candidate_entity != []:
                        end_entity = candidate_entity[0]     # 取第一个候选实体
                        candidate_entity.remove(end_entity) # 从候选列表中移除
                        
                        # 查找起始实体到目标实体的路径
                        paths,exist_entity = find_shortest_path(start_entity, end_entity, candidate_entity)
                        path_list = []
                        
                        # 如果没找到路径，需要重新选择起始实体
                        if paths == [''] or paths == []:
                            flag = 1
                            if candidate_entity == []:
                                flag = 0
                                break
                            start_entity = candidate_entity[0]
                            candidate_entity.remove(start_entity)
                            break
                        else:
                            # 解析路径字符串
                            for p in paths:
                                path_list.append(p.split('->'))
                            if path_list != []:
                                paths_list.append(path_list)
                        
                        # 如果路径中包含其他候选实体，从候选列表中移除
                        if exist_entity != {}:
                            try:
                                candidate_entity.remove(exist_entity)
                            except:
                                continue
                        start_entity = end_entity  # 更新起始实体
                    
                    # 生成路径组合
                    result_path = combine_lists(*paths_list)
                
                    # 收集所有路径
                    if result_path != []:
                        result_path_list.extend(result_path)                
                    if flag == 1:
                        continue  # 重新开始路径查找
                    else:
                        break     # 路径查找完成
                    
                # ==================== 路径后处理 ====================
                
                # 统计不同起始实体的数量
                start_tmp = []
                for path_new in result_path_list:
                    if path_new == []:
                        continue
                    if path_new[0] not in start_tmp:
                        start_tmp.append(path_new[0])
                
                # 根据起始实体数量选择路径
                if len(start_tmp) == 0:
                    result_path = {}      # 没有有效路径
                    single_path = {}
                else:
                    if len(start_tmp) == 1:
                        # 只有一个起始实体，取前5条路径
                        result_path = result_path_list[:5]
                    else:
                        # 多个起始实体，平均分配路径数量
                        result_path = []
                                                    
                        if len(start_tmp) >= 5:
                            # 起始实体数量>=5，每个取一条
                            for path_new in result_path_list:
                                if path_new == []:
                                    continue
                                if path_new[0] in start_tmp:
                                    result_path.append(path_new)
                                    start_tmp.remove(path_new[0])
                                if len(result_path) == 5:
                                    break
                        else:
                            # 起始实体数量<5，平均分配
                            count = 5 // len(start_tmp)     # 每个起始实体的路径数
                            remind = 5 % len(start_tmp)     # 余数
                            count_tmp = 0
                            for path_new in result_path_list:
                                if len(result_path) < 5:
                                    if path_new == []:
                                        continue
                                    if path_new[0] in start_tmp:
                                        if count_tmp < count:
                                            result_path.append(path_new)
                                            count_tmp += 1
                                        else:
                                            start_tmp.remove(path_new[0])
                                            count_tmp = 0
                                            if path_new[0] in start_tmp:
                                                result_path.append(path_new)
                                                count_tmp += 1

                                        # 最后一个起始实体加上余数
                                        if len(start_tmp) == 1:
                                            count = count + remind
                                else:
                                    break

                    # 记录第一条路径作为单一路径示例
                    try:
                        single_path = result_path_list[0]
                    except:
                        single_path = result_path_list
                    
            else:
                # 只匹配到一个实体或没有匹配到实体
                result_path = {}
                single_path = {}            
            
            # ==================== 邻居实体查找 ====================

            neighbor_list = []         # 存储邻居信息
            neighbor_list_disease = [] # 存储疾病相关邻居
            
            # 对每个匹配的实体查找邻居
            for match_entity in match_kg:
                disease_flag = 0  # 非疾病模式
                neighbors,disease = get_entity_neighbors(match_entity, disease_flag)
                neighbor_list.extend(neighbors)

                # 处理疾病相关实体
                while disease != []:
                    new_disease = []
                    # 检查疾病实体是否在匹配列表中
                    for disease_tmp in disease:
                        if disease_tmp in match_kg:
                            new_disease.append(disease_tmp)

                    if len(new_disease) != 0:
                        # 处理匹配列表中的疾病实体
                        for disease_entity in new_disease:
                            disease_flag = 1  # 疾病模式
                            print(disease_entity)
                            neighbors,disease = get_entity_neighbors(disease_entity, disease_flag)
                            neighbor_list_disease.extend(neighbors)
                    else:
                        # 处理其他疾病实体
                        for disease_entity in disease:
                            disease_flag = 1  # 疾病模式
                            neighbors,disease = get_entity_neighbors(disease_entity, disease_flag)
                            neighbor_list_disease.extend(neighbors)
                    
                    # 限制疾病邻居数量
                    if len(neighbor_list_disease) > 10:
                        break
            
            # 如果普通邻居较少，补充疾病邻居
            if len(neighbor_list)<=5:
                neighbor_list.extend(neighbor_list_disease)

            # ==================== 路径信息处理 ====================

            # 如果找到了路径信息
            if len(match_kg) != 1 or 0:
                response_of_KG_list_path = []
                if result_path == {}:
                    response_of_KG_list_path = []
                    path_sampled = []
                else:
                    # 格式化路径信息
                    result_new_path = []
                    for total_path_i in result_path:
                        path_input = "->".join(total_path_i)
                        result_new_path.append(path_input)
                    
                    path = "\n".join(result_new_path)
                    # 使用LLM重排序路径信息
                    path_sampled = self_knowledge_retrieval_reranking(path, input_text[0])
                    
                    # 将路径转换为自然语言
                    response_of_KG_list_path = prompt_path_finding(path_sampled)
                    # 检查回答质量，如果质量不好则重新生成
                    if is_unable_to_answer(response_of_KG_list_path):
                        response_of_KG_list_path = prompt_path_finding(path_sampled)
            else:
                response_of_KG_list_path = '{}'

            # 处理单一路径
            response_single_path = prompt_path_finding(single_path)
            if is_unable_to_answer(response_single_path):
                response_single_path = prompt_path_finding(single_path)

            # ==================== 邻居信息处理 ====================
            
            response_of_KG_list_neighbor = []
            neighbor_new_list = []
            
            # 格式化邻居信息
            for neighbor_i in neighbor_list:
                neighbor = "->".join(neighbor_i)
                neighbor_new_list.append(neighbor)

            # 限制邻居数量（最多5个）
            if len(neighbor_new_list) > 5:
                neighbor_input = "\n".join(neighbor_new_list[:5])
            else:
                neighbor_input = "\n".join(neighbor_new_list)
            
            # 使用LLM重排序邻居信息
            neighbor_input_sampled = self_knowledge_retrieval_reranking(neighbor_input, input_text[0])
            # 将邻居信息转换为自然语言
            response_of_KG_neighbor = prompt_neighbor(neighbor_input_sampled)
            if is_unable_to_answer(response_of_KG_neighbor):
                response_of_KG_neighbor = prompt_neighbor(neighbor_input_sampled)

            # ==================== 最终答案生成 ====================

            # 结合路径和邻居信息生成最终答案
            output_all = final_answer(input_text[0], response_of_KG_list_path, response_of_KG_neighbor)
            if is_unable_to_answer(output_all):
                output_all = final_answer(input_text[0], response_of_KG_list_path, response_of_KG_neighbor)

            # ==================== 结果解析和保存 ====================

            # 解析答案并计算准确率
            ret_parsed, acc_item = processor.parse(output_all, item)
            
            # 保存详细信息到结果中
            ret_parsed['path'] = path_sampled                            # 重排序后的路径
            ret_parsed['neighbor_input'] = neighbor_input_sampled        # 重排序后的邻居
            ret_parsed['response_of_KG_list_path'] = response_of_KG_list_path  # 路径的自然语言描述
            ret_parsed['response_of_KG_neighbor'] = response_of_KG_neighbor     # 邻居的自然语言描述
            
            # 统计准确率（只统计有效预测）
            if ret_parsed['prediction'] in processor.num2answer.values():
                acc += acc_item
                total_num += 1
            generated_data.append(ret_parsed)

        # ==================== 结果输出和保存 ====================

        # 打印当前数据集的结果
        print(dataset)
        print('accuracy:', acc/total_num)

        # 保存结果到JSON文件
        with open(os.path.join('./Alzheimers/result_chatgpt_mindmap', f"{dataset}_reranking.json"), 'w') as f:
            json.dump(generated_data, fp=f)