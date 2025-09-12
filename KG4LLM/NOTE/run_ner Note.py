# ===== 导入依赖模块 =====
from dataset_utils import *
# 导入之前分析的数据集处理器模块
# 包含：medmcqaZeroshotsProcessor, medqaZeroshotsProcessor, mmluZeroshotsProcessor, qa4mreZeroshotsProcessor

import json          # JSON文件读写操作
import os            # 文件系统操作（路径处理、文件检查等）
from tqdm import tqdm # 进度条显示库，用于显示循环处理进度
import time          # 时间操作，用于API调用间隔控制
import argparse      # 命令行参数解析（虽然代码中没有实际使用）
import openai        # OpenAI API客户端库，用于调用GPT模型

# ===== OpenAI API 配置 =====
api_key = ''#replace this to your key
# 存储OpenAI API密钥的变量，需要用户替换为实际的API密钥
openai.api_key=api_key
# 设置OpenAI库的API密钥，用于后续API调用认证

# ===== 数据集名称到处理器类的映射字典 =====
dataset2processor = {
    'medmcqa': medmcqaZeroshotsProcessor,    # MedMCQA数据集 → MedMCQA处理器类
    'medqa': medqaZeroshotsProcessor,        # MedQA数据集 → MedQA处理器类  
    'mmlu': mmluZeroshotsProcessor,          # MMLU数据集 → MMLU处理器类
    'qa4mre':qa4mreZeroshotsProcessor        # QA4MRE数据集 → QA4MRE处理器类
}
# 这个字典的作用：根据数据集名称字符串，快速获取对应的处理器类

# ===== API调用函数 =====
def request_api_chatgpt(prompt):
    """
    调用OpenAI ChatGPT API进行NER任务
    
    参数：
        prompt: 发送给GPT的提示词
    
    返回：
        GPT返回的文本结果
    """
    
    # 构建消息格式，符合OpenAI Chat API要求
    messages = [
        # 系统消息：设定AI助手的角色和专业领域
        {"role": "system", "content": 'You are an AI assistant to answer question about biomedicine.'},
        # 用户消息：实际的NER提示词
        {"role": "user", "content": prompt}
    ]
    
    try:
        # 调用OpenAI ChatCompletion API
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",    # 使用GPT-3.5-turbo模型
            messages=messages,        # 传入构建的消息
        )
        # 提取AI返回的文本内容并去除首尾空白
        ret = completion["choices"][0]["message"]["content"].strip()
        return ret  # 返回处理后的结果
        
    except Exception as E:
        # 异常处理：如果API调用失败
        time.sleep(2)  # 等待2秒，避免频繁重试
        print(E)       # 打印错误信息
        return request_api_chatgpt(prompt)  # 递归重试，直到成功

# ===== 主函数 =====
def main():
    """
    主要处理流程：
    1. 遍历四个医学数据集
    2. 对每个数据集进行NER处理
    3. 保存带有实体信息的新数据文件
    """
    
    # 遍历所有需要处理的数据集
    for dataset in ['medqa', 'medmcqa', 'mmlu', 'qa4mre']:
        
        # ===== 步骤1：初始化数据集处理器 =====
        processor = dataset2processor[dataset]()
        # 根据数据集名称获取对应的处理器类，并实例化
        # 例如：dataset='medqa' → medqaZeroshotsProcessor()
        
        # ===== 步骤2：加载原始数据 =====
        data = processor.load_original_dataset()
        # 加载原始过滤后的数据（不包含entity字段）
        # 从 'Alzheimers/result_filter/{dataset}_filter.json' 加载
        
        # ===== 步骤3：初始化处理变量 =====
        generated_data = []  # 存储处理后的数据（包含entity字段）
        acc, total_num = 0, 0  # 准确率和总数统计（这里似乎没有实际使用）
        
        # ===== 步骤4：遍历数据集中的每个问题 =====
        for item in tqdm(data):
            # tqdm(data)：为循环添加进度条显示
            # item：单个问题数据，包含question, opa, opb, opc, opd, cop等字段
            
            # ===== API调用频率控制 =====
            time.sleep(2)
            # 每次API调用前等待2秒，避免触发OpenAI的频率限制
            
            # ===== 步骤5：生成NER提示词 =====
            prompt = processor.generate_prompt_ner(item)
            # 调用处理器的generate_prompt_ner方法
            # 生成类似这样的提示：
            # "Extract all the biomedicine-related entity from the following question and choices..."
            
            # ===== 步骤6：调用GPT进行NER处理 =====
            ret = request_api_chatgpt(prompt)
            # 将NER提示发送给GPT-3.5，获取提取的实体列表
            # 返回类似："1. Alzheimer's disease\n2. Dementia\n3. ..."
            
            # ===== 步骤7：将实体结果添加到数据项 =====
            item['entity'] = ret
            # 在原始数据项中添加'entity'字段，存储GPT提取的实体
            # 这样原始数据就变成了带有实体信息的数据
            
            # ===== 步骤8：保存到结果列表 =====
            generated_data.append(item)
            # 将处理后的数据项添加到结果列表中
        
        # ===== 步骤9：保存处理后的数据到文件 =====
        with open(os.path.join('Alzheimers', 'result_ner', f"{dataset}_zero-shot.json"), 'w') as f:
            # 构建保存路径：'Alzheimers/result_ner/{dataset}_zero-shot.json'
            # 例如：'Alzheimers/result_ner/medqa_zero-shot.json'
            json.dump(generated_data, fp=f)
            # 将包含entity字段的完整数据保存为JSON文件

# ===== 程序入口 =====
if __name__ == '__main__':
    main()
    # 如果直接运行此脚本，执行main函数
    # 开始处理所有四个数据集的NER任务