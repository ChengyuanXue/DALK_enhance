# ====================================================================
# DALK项目 - API工具函数模块
# 用于调用Google PaLM API和OpenAI ChatGPT API的工具函数
# 这个模块是LLM4KG部分的核心，用于从科学文献中提取阿尔兹海默病知识图谱
# ====================================================================

import time  # 导入time模块，用于添加延时防止API调用过于频繁
import openai  # 导入OpenAI官方Python库，用于调用GPT模型
import google.generativeai as palm  # 导入Google的生成式AI库，用于调用PaLM模型
from google.generativeai.types import safety_types  # 导入安全设置相关的类型定义
from google.api_core import retry  # 导入Google API核心库的重试装饰器

# ====================================================================
# API密钥配置部分
# ====================================================================

# 配置Google PaLM API密钥
# 用户需要在Google AI Studio获取API密钥并替换空字符串
palm.configure(api_key='')  # 替换为你的Google PaLM API密钥

# OpenAI API密钥变量
# 用户需要在OpenAI平台获取API密钥并替换空字符串  
api_key = ''  # 替换为你的OpenAI API密钥

# 设置OpenAI库的全局API密钥
# 这样后续调用openai.ChatCompletion.create时就不需要再传入api_key参数
openai.api_key = api_key

# ====================================================================
# Google PaLM API调用函数
# ====================================================================

@retry.Retry()  # 使用Google API核心库的重试装饰器，自动处理网络错误和临时故障
def request_api_palm(messages):
    """
    调用Google PaLM API生成文本的函数
    
    参数:
        messages (str): 输入的提示词文本
        
    返回:
        str: PaLM模型生成的响应文本
        
    功能:
        - 使用text-bison-001模型进行文本生成
        - 配置了宽松的安全设置以避免内容被过度审查
        - 自动重试机制处理API调用失败
    """
    
    # 指定使用的PaLM模型版本
    # text-bison-001是Google的文本生成模型，适用于各种NLP任务
    model = 'models/text-bison-001'
    
    # 调用PaLM API生成文本
    completion = palm.generate_text(
        model=model,  # 指定模型
        prompt=messages,  # 输入的提示词
        
        # 安全设置配置 - 设置为最宽松的级别
        # 这对于医学文献处理很重要，因为医学内容可能被误判为敏感内容
        safety_settings=[
            {
                # 贬损性内容检测 - 设置为不阻止
                "category": safety_types.HarmCategory.HARM_CATEGORY_DEROGATORY,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                # 暴力内容检测 - 设置为不阻止  
                "category": safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                # 毒性内容检测 - 设置为不阻止
                "category": safety_types.HarmCategory.HARM_CATEGORY_TOXICITY,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                # 医疗内容检测 - 设置为不阻止
                # 这个设置对于处理阿尔兹海默病相关医学文献特别重要
                "category": safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
        ]
    )
    
    # 检查API响应是否包含有效结果
    # 如果candidates列表为空，说明内容被安全过滤器阻止或其他错误
    if len(completion.candidates) < 1:
        print(completion)  # 打印完整响应以便调试
    
    # 提取并返回生成的文本内容
    # candidates[0]['output']包含模型生成的主要文本输出
    ret = completion.candidates[0]['output']
    return ret

    # 输出的样子：
    #     completion = {
    #     'candidates': [
    #         {
    #             'output': '这里是模型生成的文本内容',
    #             'safety_ratings': [...],
    #             'finish_reason': 'STOP'
    #         },
    #         # 可能有多个候选结果
    #     ],
    #     'filters': [...],
    #     'safety_feedback': [...]
    # }

# ====================================================================
# OpenAI ChatGPT API调用函数  
# ====================================================================

def request_api_chatgpt(prompt):
    """
    调用OpenAI ChatGPT API生成响应的函数
    
    参数:
        prompt (str): 用户输入的提示词
        
    返回:
        str: ChatGPT模型生成的响应文本
        
    功能:
        - 使用gpt-3.5-turbo模型进行对话生成
        - 设置系统角色为生物医学问答助手
        - 包含错误处理和自动重试机制
    """
    
    # 构建ChatGPT API所需的消息格式
    # ChatGPT使用对话格式，需要指定角色和内容
    messages = [
        {
            "role": "system",  # 系统角色 - 定义AI助手的行为和专业领域
            "content": 'You are an AI assistant to answer question about biomedicine.'  # 定义为生物医学问答助手
        },
        {
            "role": "user",    # 用户角色 - 用户的输入
            "content": prompt  # 用户的具体问题或提示
        }
    ]
    
    try:
        # 调用OpenAI ChatCompletion API
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 使用GPT-3.5 Turbo模型，性价比高且响应快
            messages=messages,       # 传入构建好的消息列表
        )
        
        # 从API响应中提取生成的文本内容
        # choices[0]获取第一个（通常是唯一的）生成结果
        # message.content包含实际的响应文本
        # strip()去除首尾空白字符
        ret = completion["choices"][0]["message"]["content"].strip()
        return ret
        
    except Exception as E:
        # 异常处理：API调用失败时的处理逻辑
        time.sleep(2)  # 等待2秒，避免过于频繁的重试请求
        print(E)       # 打印错误信息用于调试
        
        # 递归调用自身进行重试
        # 这是一个简单的重试机制，会一直重试直到成功
        return request_api_chatgpt(prompt)


# ChatGPT API响应的完整数据结构：
#     completion = {
#         'id': 'chatcmpl-8abc123def456ghi789',  # 请求的唯一标识符
#         'object': 'chat.completion',            # 对象类型
#         'created': 1677652288,                  # 创建时间戳
#         'model': 'gpt-3.5-turbo-0613',         # 实际使用的模型版本
#         'choices': [                            # 生成的选择列表
#             {
#                 'index': 0,                     # 选择的索引
#                 'message': {                    # 消息对象
#                     'role': 'assistant',        # 角色：助手
#                     'content': '这里是ChatGPT生成的实际回答文本内容...'  # 实际生成的文本
#                 },
#                 'finish_reason': 'stop'         # 结束原因：stop(正常结束)、length(达到长度限制)、content_filter(内容过滤)
#             }
#         ],
#         'usage': {                              # 使用情况统计
#             'prompt_tokens': 57,                # 输入提示的token数量
#             'completion_tokens': 40,            # 生成回答的token数量  
#             'total_tokens': 97                  # 总token数量
#         },
#         'system_fingerprint': 'fp_44709d6fcb'  # 系统指纹标识
#     }

# 提取内容的路径：
# completion["choices"][0]["message"]["content"]
#    └── choices列表的第一个元素
#            └── message对象  
#                    └── content字段 ← 这里是实际的文本内容

# 可能的finish_reason值：
# - 'stop': 模型自然结束生成
# - 'length': 达到最大token限制
# - 'content_filter': 内容被安全过滤器阻止
# - 'null': 生成仍在进行中（流式响应时）

# 与PaLM API的对比：
# PaLM:     completion.candidates[0]['output']
# ChatGPT:  completion["choices"][0]["message"]["content"]