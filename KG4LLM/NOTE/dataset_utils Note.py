# ====================================================================
# DALK项目 - 多选题问答数据集处理器模块
# 目的：为不同的医学问答数据集提供统一的数据处理接口
# 支持的数据集：MedMCQA, MedQA, MMLU, QA4MRE
# 功能：NER提示生成、问答提示生成、结果解析、准确率计算
# ====================================================================

import datasets   # Hugging Face datasets库
import random     # 随机数生成，用于数据采样
random.seed(42)   # 设置随机种子，确保结果可重现
import json       # JSON文件处理
# from api_utils import *  # API工具函数（已注释）
import time       # 时间处理
import os         # 文件系统操作

# ====================================================================
# 基础处理器类 - 定义通用的提示模板和接口
# ====================================================================
class Processor:
    def __init__(self):
        # ============================================================
        # 命名实体识别(NER)提示模板
        # 目的：从问题和选项中提取生物医学相关的实体
        # 翻译：从以下问题和选项中提取所有与生物医学相关的实体，将每个实体单独列在一行，并用序号标记（1., 2., ...）
        #   问题：{}
        #   提取的实体有：
        # ============================================================
        self.template_ner = '''Extract all the biomedicine-related entity from the following question and choices, output each entity in a single line with a serial number (1., 2., ...)
Question: {}
The extracted entities are:
'''
        
        # ============================================================
        # 基础问答提示模板
        # 目的：直接进行问答，不使用推理链
        # 翻译：问题：{}
        #     答案：选项是：
        # ============================================================
        self.template = '''Question: {} 
Answer: The option is: '''
        
        # ============================================================
        # Chain-of-Thought (CoT) 推理提示模板
        # 目的：引导LLM进行逐步推理
        # 翻译：问题：{}
        #     答案：让我们一步一步地思考。
        # ============================================================
        self.template_CoT = '''Question: {} 
Answer: Let's think step by step. '''
        
        # ============================================================
        # 推理结果整合模板
        # 目的：基于CoT推理结果，提取最终答案
        # 翻译：问题：{}
        #     答案：让我们一步一步地思考。{} 因此，字母选项（仅填字母）是：
        # ============================================================
        self.template_inference = '''Question: {} 
Answer: Let's think step by step. {} Therefore, the letter option (only the letter) is:'''

    def load_dataset(self):
        """加载处理后的数据集（包含NER结果）"""
        return self.data

    def load_original_dataset(self):
        """加载原始过滤后的数据集"""
        return self.data_original

# ====================================================================
# MedMCQA数据集处理器
# MedMCQA：印度医学院入学考试题目数据集
# ====================================================================
class medmcqaZeroshotsProcessor(Processor):
    def __init__(self):
        super().__init__()
        
        # 尝试加载已处理的NER结果文件
        if os.path.exists(os.path.join('Alzheimers','result_ner', 'medmcqa_zero-shot.json')):
            self.data = json.load(open(os.path.join('Alzheimers','result_ner', 'medmcqa_zero-shot.json')))
        
        # 加载原始过滤后的数据
        self.data_original = json.load(open(os.path.join('Alzheimers', 'result_filter', 'medmcqa_filter.json')))
        
        # 数字答案到字母选项的映射
        self.num2answer = {
            0: 'A',  # 第一个选项对应A
            1: 'B',  # 第二个选项对应B
            2: 'C',  # 第三个选项对应C
            3: 'D'   # 第四个选项对应D
        }

    def generate_prompt_ner(self, item):
        """
        生成NER任务的提示
        
        参数:
        item: 数据项，包含问题和选项
        
        返回:
        格式化的NER提示字符串
        """
        # 提取问题文本
        question = item['question']
        
        # 提取四个选项（MedMCQA格式）
        A, B, C, D = item['opa'], item['opb'], item['opc'], item['opd']
        
        # 格式化选项为标准多选题格式
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        
        # 将问题和选项合并
        question += option

        # 使用NER模板生成提示
        prompt_ner = self.template_ner.format(question)
        return prompt_ner

    def generate_prompt(self, item):
        """
        生成问答任务的提示
        
        参数:
        item: 数据项
        
        返回:
        格式化的问题字符串（包含选项）
        """
        # 提取问题文本
        question = item['question']
        
        # 提取四个选项
        A, B, C, D = item['opa'], item['opb'], item['opc'], item['opd']
        
        # 格式化选项
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        
        # 返回完整的问题（问题+选项）
        question += option
        return question

    def parse(self, ret, item):
        """
        解析LLM返回结果并计算准确率
        
        参数:
        ret: LLM的返回结果
        item: 原始数据项
        
        返回:
        (updated_item, accuracy): 更新后的数据项和准确率
        """
        # 清理返回结果，移除句号
        ret = ret.replace('.', '')
        
        # 如果返回结果长度大于1，只取第一个字符
        if len(ret) > 1:
            ret = ret[0]
        
        # 将预测结果添加到数据项中
        item['prediction'] = ret
        
        # 获取正确答案（MedMCQA中cop字段存储正确答案的索引）
        answer = item['cop']
        answer = self.num2answer[answer]  # 转换为字母选项
        
        # 计算准确率
        if answer.strip() == ret.strip():
            acc = 1  # 预测正确
        else:
            acc = 0  # 预测错误
        
        return item, acc

# ====================================================================
# MedQA数据集处理器
# MedQA：美国医师执业考试(USMLE)风格的题目
# ====================================================================
class medqaZeroshotsProcessor(Processor):
    def __init__(self):
        super().__init__()
        
        # 加载NER结果文件
        if os.path.exists(os.path.join('Alzheimers','result_ner', 'medqa_zero-shot.json')):
            self.data = json.load(open(os.path.join('Alzheimers','result_ner', 'medqa_zero-shot.json')))
        
        # 加载原始数据
        self.data_original = json.load(open(os.path.join('Alzheimers', 'result_filter', 'medqa_filter.json')))
        
        # 答案映射
        self.num2answer = {
            0: 'A',
            1: 'B', 
            2: 'C',
            3: 'D'
        }

    def generate_prompt_ner(self, item):
        """生成MedQA的NER提示"""
        question = item['question']
        
        # MedQA的选项存储在choices列表中
        A, B, C, D = item['choices'][0], item['choices'][1], item['choices'][2], item['choices'][3]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option

        prompt_ner = self.template_ner.format(question)
        return prompt_ner

    def generate_prompt(self, item):
        """生成MedQA的问答提示"""
        question = item['question']
        A, B, C, D = item['choices'][0], item['choices'][1], item['choices'][2], item['choices'][3]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option
        return question

    def parse(self, ret, item):
        """解析MedQA的结果"""
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]
        item['prediction'] = ret
        
        # MedQA的答案存储格式不同
        answer = item['answer'][0]  # 答案是列表，取第一个元素
        answer = item['choices'].index(answer)  # 找到答案在choices中的索引
        answer = self.num2answer[answer]  # 转换为字母
        
        if answer.strip() == ret.strip():
            acc = 1
        else:
            acc = 0
        return item, acc

# ====================================================================
# MMLU数据集处理器  
# MMLU：大规模多任务语言理解基准测试
# ====================================================================
class mmluZeroshotsProcessor(Processor):
    def __init__(self):
        super().__init__()
        
        # 加载数据文件
        if os.path.exists(os.path.join('Alzheimers','result_ner', 'mmlu_zero-shot.json')):
            self.data = json.load(open(os.path.join('Alzheimers','result_ner', 'mmlu_zero-shot.json')))
        self.data_original = json.load(open(os.path.join('Alzheimers', 'result_filter', 'mmlu_filter.json')))
        
        # 答案映射
        self.num2answer = {
            0: 'A',
            1: 'B',
            2: 'C', 
            3: 'D'
        }

    def generate_prompt_ner(self, item):
        """生成MMLU的NER提示"""
        question = item['question']
        A, B, C, D = item['choices'][0], item['choices'][1], item['choices'][2], item['choices'][3]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option

        prompt_ner = self.template_ner.format(question)
        return prompt_ner

    def generate_prompt(self, item):
        """生成MMLU的问答提示"""
        question = item['question']
        A, B, C, D = item['choices'][0], item['choices'][1], item['choices'][2], item['choices'][3]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option
        return question

    def parse(self, ret, item):
        """解析MMLU的结果"""
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]
        item['prediction'] = ret
        
        # MMLU的答案直接是数字索引
        answer = item['answer']
        answer = self.num2answer[answer]
        
        if answer.strip() == ret.strip():
            acc = 1
        else:
            acc = 0
        return item, acc

# ====================================================================
# QA4MRE数据集处理器
# QA4MRE：机器阅读理解评测的问答数据集
# ====================================================================
class qa4mreZeroshotsProcessor(Processor):
    def __init__(self):
        super().__init__()
        
        # 加载数据文件
        if os.path.exists(os.path.join('Alzheimers','result_ner', 'qa4mre_zero-shot.json')):
            self.data = json.load(open(os.path.join('Alzheimers','result_ner', 'qa4mre_zero-shot.json')))
        self.data_original = json.load(open(os.path.join('Alzheimers', 'result_filter', 'qa4mre_filter.json')))
        
        # QA4MRE有5个选项，答案ID从1开始
        self.num2answer = {
            1: 'A',
            2: 'B',
            3: 'C',
            4: 'D',
            5: 'E'
        }

    def generate_prompt_ner(self, item):
        """生成QA4MRE的NER提示"""
        # QA4MRE的问题字段是question_str
        question = item['question_str']
        
        # QA4MRE有5个选项
        A, B, C, D, E = item['answer_options']['answer_str'][0], item['answer_options']['answer_str'][1], item['answer_options']['answer_str'][2], item['answer_options']['answer_str'][3], item['answer_options']['answer_str'][4]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'+'E.'+E+'\n'
        question += option

        prompt_ner = self.template_ner.format(question)
        return prompt_ner

    def generate_prompt(self, item):
        """生成QA4MRE的问答提示"""
        question = item['question_str']
        A, B, C, D, E = item['answer_options']['answer_str'][0], item['answer_options']['answer_str'][1], item['answer_options']['answer_str'][2], item['answer_options']['answer_str'][3], item['answer_options']['answer_str'][4]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'+'E.'+E+'\n'
        question += option
        return question

    def parse(self, ret, item):
        """解析QA4MRE的结果"""
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]
        item['prediction'] = ret
        
        # QA4MRE的正确答案ID
        answer = item['correct_answer_id']
        answer = self.num2answer[int(answer)]  # 转换为字母
        
        if answer.strip() == ret.strip():
            acc = 1
        else:
            acc = 0
        return item, acc

# ====================================================================
# 使用说明
# ====================================================================
# 
# 这些处理器类用于：
# 1. 统一不同数据集的数据格式和接口
# 2. 生成用于NER任务的提示
# 3. 生成用于问答任务的提示
# 4. 解析LLM的输出并计算准确率
#
# 使用示例：
# processor = medmcqaZeroshotsProcessor()
# dataset = processor.load_dataset()
# for item in dataset:
#     prompt = processor.generate_prompt(item)
#     # 调用LLM获取结果
#     result, acc = processor.parse(llm_output, item)
#
# 文件结构：
# Alzheimers/
# ├── result_filter/          # 过滤后的原始数据集
# │   ├── medmcqa_filter.json
# │   ├── medqa_filter.json
# │   ├── mmlu_filter.json
# │   └── qa4mre_filter.json
# └── result_ner/             # NER处理后的数据集
#     ├── medmcqa_zero-shot.json
#     ├── medqa_zero-shot.json
#     ├── mmlu_zero-shot.json
#     └── qa4mre_zero-shot.json


# # MedMCQA数据处理详细流程示例

# ## 输入数据样本

# ```json
# {
#   "id": "fd8ef88d-5c1d-408a-8821-22c2ad3a590f", 
#   "question": "Alzheimer's disease is associated with: September 2012", 
#   "opa": "Delerium", 
#   "opb": "Delusion", 
#   "opc": "Dementia", 
#   "opd": "All of the above", 
#   "cop": 2, 
#   "choice_type": "multi", 
#   "exp": "Ans. C i.e. Dementia...", 
#   "subject_name": "Psychiatry", 
#   "topic_name": null,
#   "entity": "1. Alzheimer's disease\n2. Delirium \n3. Delusion \n4. Dementia"
# }
# ```

# ## 处理流程详解

# ### 步骤1: 初始化处理器

# ```python
# class medmcqaZeroshotsProcessor(Processor):
#     def __init__(self):
#         super().__init__()
        
#         # 加载已处理的NER结果文件（如果存在）
#         if os.path.exists(os.path.join('Alzheimers','result_ner', 'medmcqa_zero-shot.json')):
#             self.data = json.load(open(os.path.join('Alzheimers','result_ner', 'medmcqa_zero-shot.json')))
        
#         # 加载原始过滤后的数据
#         self.data_original = json.load(open(os.path.join('Alzheimers', 'result_filter', 'medmcqa_filter.json')))
        
#         # 数字答案到字母选项的映射
#         self.num2answer = {
#             0: 'A',  # opa对应A
#             1: 'B',  # opb对应B  
#             2: 'C',  # opc对应C
#             3: 'D'   # opd对应D
#         }
# ```

# ### 步骤2: NER提示生成

# ```python
# def generate_prompt_ner(self, item):
#     # 提取问题文本
#     question = item['question']
#     # question = "Alzheimer's disease is associated with: September 2012"
    
#     # 提取四个选项
#     A = item['opa']  # "Delerium"
#     B = item['opb']  # "Delusion" 
#     C = item['opc']  # "Dementia"
#     D = item['opd']  # "All of the above"
    
#     # 格式化选项为标准多选题格式
#     option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
#     # option = "\nA.Delerium\nB.Delusion\nC.Dementia\nD.All of the above\n"
    
#     # 将问题和选项合并
#     question += option
#     # question = "Alzheimer's disease is associated with: September 2012\nA.Delerium\nB.Delusion\nC.Dementia\nD.All of the above\n"

#     # 使用NER模板生成提示
#     prompt_ner = self.template_ner.format(question)
#     return prompt_ner
# ```

# **生成的NER提示完整内容：**
# ```
# Extract all the biomedicine-related entity from the following question and choices, output each entity in a single line with a serial number (1., 2., ...)
# Question: Alzheimer's disease is associated with: September 2012
# A.Delerium
# B.Delusion
# C.Dementia
# D.All of the above

# The extracted entities are:
# ```

# **对应的实际NER结果（来自数据中的entity字段）：**
# ```
# 1. Alzheimer's disease
# 2. Delirium 
# 3. Delusion 
# 4. Dementia
# ```

# ### 步骤3: 问答提示生成

# ```python
# def generate_prompt(self, item):
#     # 提取问题文本
#     question = item['question']
#     # question = "Alzheimer's disease is associated with: September 2012"
    
#     # 提取四个选项
#     A = item['opa']  # "Delerium"
#     B = item['opb']  # "Delusion"
#     C = item['opc']  # "Dementia" 
#     D = item['opd']  # "All of the above"
    
#     # 格式化选项
#     option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
    
#     # 返回完整的问题（问题+选项）
#     question += option
#     return question
# ```

# **生成的问答提示完整内容：**
# ```
# Alzheimer's disease is associated with: September 2012
# A.Delerium
# B.Delusion
# C.Dementia
# D.All of the above
# ```

# ### 步骤4: 使用不同模板的完整提示

# **基础模板 (self.template)：**
# ```
# Question: Alzheimer's disease is associated with: September 2012
# A.Delerium
# B.Delusion
# C.Dementia
# D.All of the above
 
# Answer: The option is: 
# ```

# **CoT模板 (self.template_CoT)：**
# ```
# Question: Alzheimer's disease is associated with: September 2012
# A.Delerium
# B.Delusion
# C.Dementia
# D.All of the above
 
# Answer: Let's think step by step. 
# ```

# ### 步骤5: 结果解析

# ```python
# def parse(self, ret, item):
#     # 假设LLM返回的结果是 "C" 或 "C." 
#     ret = "C."
    
#     # 清理返回结果，移除句号
#     ret = ret.replace('.', '')
#     # ret = "C"
    
#     # 如果返回结果长度大于1，只取第一个字符
#     if len(ret) > 1:
#         ret = ret[0]
#     # ret = "C" (长度为1，不变)
    
#     # 将预测结果添加到数据项中
#     item['prediction'] = ret
#     # item['prediction'] = "C"
    
#     # 获取正确答案
#     answer = item['cop']  # 2
#     answer = self.num2answer[answer]  # self.num2answer[2] = "C"
    
#     # 计算准确率
#     if answer.strip() == ret.strip():  # "C" == "C"
#         acc = 1  # 预测正确
#     else:
#         acc = 0  # 预测错误
    
#     return item, acc  # 返回更新后的item和准确率1
# ```

# **更新后的数据项：**
# ```json
# {
#   "id": "fd8ef88d-5c1d-408a-8821-22c2ad3a590f", 
#   "question": "Alzheimer's disease is associated with: September 2012", 
#   "opa": "Delerium", 
#   "opb": "Delusion", 
#   "opc": "Dementia", 
#   "opd": "All of the above", 
#   "cop": 2,
#   "prediction": "C",
#   "choice_type": "multi", 
#   "exp": "Ans. C i.e. Dementia...", 
#   "subject_name": "Psychiatry", 
#   "topic_name": null,
#   "entity": "1. Alzheimer's disease\n2. Delirium \n3. Delusion \n4. Dementia"
# }
# ```

# ## 数据字段含义解析

# - **id**: 题目唯一标识符
# - **question**: 问题文本 
# - **opa, opb, opc, opd**: 四个选项 (option a, b, c, d)
# - **cop**: 正确答案索引 (correct option) - 0=A, 1=B, 2=C, 3=D
# - **choice_type**: 选择题类型
# - **exp**: 题目解析说明
# - **subject_name**: 学科名称
# - **topic_name**: 主题名称  
# - **entity**: NER提取的实体列表
# - **prediction**: 模型预测的答案 (处理后添加)

# ## 核心处理逻辑

# 1. **数据标准化**: 将不同格式的选项统一为A、B、C、D格式
# 2. **提示模板化**: 使用预定义模板生成标准化的提示
# 3. **答案映射**: 将数字索引转换为字母选项，便于比较
# 4. **结果清理**: 移除标点符号，确保答案格式一致
# 5. **准确率计算**: 简单的字符串匹配来判断预测是否正确

# ## 实际运行示例

# ```python
# # 初始化处理器
# processor = medmcqaZeroshotsProcessor()

# # 示例数据项
# item = {
#     "id": "fd8ef88d-5c1d-408a-8821-22c2ad3a590f", 
#     "question": "Alzheimer's disease is associated with: September 2012", 
#     "opa": "Delerium", 
#     "opb": "Delusion", 
#     "opc": "Dementia", 
#     "opd": "All of the above", 
#     "cop": 2
# }

# # 生成NER提示
# ner_prompt = processor.generate_prompt_ner(item)
# print("NER提示:", ner_prompt)

# # 生成问答提示  
# qa_prompt = processor.generate_prompt(item)
# print("问答提示:", qa_prompt)

# # 模拟LLM返回结果并解析
# llm_output = "C"  # 假设LLM返回C
# updated_item, accuracy = processor.parse(llm_output, item)
# print("预测结果:", updated_item['prediction'])
# print("准确率:", accuracy)
# ```

# 这个处理器的设计使得不同医学问答数据集能够通过统一的接口进行处理，同时保持了足够的灵活性来处理各种数据格式的差异。