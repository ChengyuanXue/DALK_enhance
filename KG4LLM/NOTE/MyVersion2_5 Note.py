# ==============================================================================
# 医学知识图谱问答系统 - 详细注释版本
# Medical Knowledge Graph Question Answering System - Detailed Comments
# ==============================================================================

# ========================= 导入依赖库 =========================
# LangChain相关库 - 用于构建LLM应用链
from langchain.chat_models import ChatOpenAI  # OpenAI聊天模型包装器
from langchain import PromptTemplate, LLMChain  # 提示模板和LLM链
from langchain.prompts.chat import (  # 聊天提示相关组件
    ChatPromptTemplate,      # 聊天提示模板
    SystemMessagePromptTemplate,  # 系统消息提示模板
    AIMessagePromptTemplate,      # AI消息提示模板
    HumanMessagePromptTemplate,   # 人类消息提示模板
)
from langchain.schema import (  # LangChain消息模式
    AIMessage,      # AI消息类型
    HumanMessage,   # 人类消息类型
    SystemMessage,  # 系统消息类型
)

# 数据处理和数值计算库
import numpy as np  # 数值计算库
import re           # 正则表达式库
import string       # 字符串处理工具
import pandas as pd # 数据分析库

# 图数据库Neo4j相关
from neo4j import GraphDatabase, basic_auth  # Neo4j数据库驱动

# 数据结构和算法工具
from collections import deque, Counter, defaultdict  # 队列、计数器、默认字典
import itertools    # 迭代工具
from typing import Dict, List, Tuple, Optional  # 类型提示

# 数据存储和序列化
import pickle  # Python对象序列化
import json    # JSON数据处理

# 机器学习和NLP库
from sklearn.metrics.pairwise import cosine_similarity  # 余弦相似度计算
from sklearn.preprocessing import normalize              # 数据标准化
import openai  # OpenAI API客户端

# NLP评估指标库
from pycocoevalcap.bleu.bleu import Bleu      # BLEU评分
from pycocoevalcap.cider.cider import Cider    # CIDEr评分
from pycocoevalcap.rouge.rouge import Rouge    # ROUGE评分
from pycocoevalcap.meteor.meteor import Meteor # METEOR评分

# LangChain LLM模型
from langchain.llms import OpenAI  # OpenAI LLM包装器

# 系统相关库
import os        # 操作系统接口
import sys       # 系统特定参数和函数
import logging   # 日志记录
import gc        # 垃圾回收
from time import sleep  # 时间延迟
from functools import wraps  # 装饰器工具
from datetime import datetime  # 日期时间处理

# 图像处理库
from PIL import Image, ImageDraw, ImageFont  # Python图像处理库

# 文件处理
import csv  # CSV文件处理

# 文本相似度和信息检索
from gensim import corpora  # Gensim语料库工具
from gensim.models import TfidfModel  # TF-IDF模型
from gensim.similarities import SparseMatrixSimilarity  # 稀疏矩阵相似度
from rank_bm25 import BM25Okapi  # BM25检索算法
from gensim.models import Word2Vec  # Word2Vec词向量模型

# 网络请求和加密
import requests      # HTTP请求库
import urllib.parse  # URL解析
import hashlib       # 哈希算法
import hmac          # HMAC认证
import base64        # Base64编码

# XML处理
import xml.etree.ElementTree as ET  # XML解析

# 进度条
from tqdm import tqdm  # 进度条显示

# 自定义数据集处理工具
from dataset_utils import *  # 导入自定义的数据集处理函数

# ========================= 消融实验配置 =========================
# 🔬 消融实验开关配置
# 消融实验(Ablation Study)用于测试系统各个组件的重要性
ABLATION_CONFIGS = {
    # 基线配置 - 所有增强功能都关闭
    'baseline': {
        'USE_HIERARCHICAL_KG': False,      # 不使用层次化知识图谱
        'USE_MULTI_STRATEGY_LINKING': False,  # 不使用多策略实体链接
        'USE_ADAPTIVE_UMLS': False,        # 不使用自适应UMLS知识选择
        'USE_UMLS_NORMALIZATION': False,   # 不使用UMLS标准化
        'USE_REASONING_RULES': False,      # 不使用推理规则
        'USE_KG_GUIDED_REASONING': False,  # 不使用知识图谱引导推理
        'USE_OPTIMIZED_MULTIHOP': False,   # 不使用优化多跳推理
        'USE_ENHANCED_ANSWER_GEN': False   # 不使用增强答案生成
    },
    # 完整模型配置 - 所有增强功能都开启
    'full_model': {
        'USE_HIERARCHICAL_KG': True,       # 使用层次化知识图谱
        'USE_MULTI_STRATEGY_LINKING': True,   # 使用多策略实体链接
        'USE_ADAPTIVE_UMLS': True,         # 使用自适应UMLS知识选择
        'USE_UMLS_NORMALIZATION': True,    # 使用UMLS标准化
        'USE_REASONING_RULES': True,       # 使用推理规则
        'USE_KG_GUIDED_REASONING': True,   # 使用知识图谱引导推理
        'USE_OPTIMIZED_MULTIHOP': True,    # 使用优化多跳推理
        'USE_ENHANCED_ANSWER_GEN': True    # 使用增强答案生成
    },
    # 以下是各种消融配置，每个配置关闭一个特定功能来测试其重要性
    'ablation_hierarchical_kg': {
        'USE_HIERARCHICAL_KG': False,      # 关闭层次化知识图谱
        'USE_MULTI_STRATEGY_LINKING': True,
        'USE_ADAPTIVE_UMLS': True,
        'USE_UMLS_NORMALIZATION': True,
        'USE_REASONING_RULES': True,
        'USE_KG_GUIDED_REASONING': True,
        'USE_OPTIMIZED_MULTIHOP': True,
        'USE_ENHANCED_ANSWER_GEN': True
    },
    'ablation_multi_strategy': {
        'USE_HIERARCHICAL_KG': True,
        'USE_MULTI_STRATEGY_LINKING': False,  # 关闭多策略链接
        'USE_ADAPTIVE_UMLS': True,
        'USE_UMLS_NORMALIZATION': True,
        'USE_REASONING_RULES': True,
        'USE_KG_GUIDED_REASONING': True,
        'USE_OPTIMIZED_MULTIHOP': True,
        'USE_ENHANCED_ANSWER_GEN': True
    },
    'ablation_adaptive_umls': {
        'USE_HIERARCHICAL_KG': True,
        'USE_MULTI_STRATEGY_LINKING': True,
        'USE_ADAPTIVE_UMLS': False,        # 关闭自适应UMLS
        'USE_UMLS_NORMALIZATION': True,
        'USE_REASONING_RULES': True,
        'USE_KG_GUIDED_REASONING': True,
        'USE_OPTIMIZED_MULTIHOP': True,
        'USE_ENHANCED_ANSWER_GEN': True
    },
    'ablation_umls_normalization': {
        'USE_HIERARCHICAL_KG': True,
        'USE_MULTI_STRATEGY_LINKING': True,
        'USE_ADAPTIVE_UMLS': True,
        'USE_UMLS_NORMALIZATION': False,   # 关闭UMLS标准化
        'USE_REASONING_RULES': True,
        'USE_KG_GUIDED_REASONING': True,
        'USE_OPTIMIZED_MULTIHOP': True,
        'USE_ENHANCED_ANSWER_GEN': True
    },
    'ablation_reasoning_rules': {
        'USE_HIERARCHICAL_KG': True,
        'USE_MULTI_STRATEGY_LINKING': True,
        'USE_ADAPTIVE_UMLS': True,
        'USE_UMLS_NORMALIZATION': True,
        'USE_REASONING_RULES': False,      # 关闭推理规则
        'USE_KG_GUIDED_REASONING': True,
        'USE_OPTIMIZED_MULTIHOP': True,
        'USE_ENHANCED_ANSWER_GEN': True
    },
    'ablation_kg_guided': {
        'USE_HIERARCHICAL_KG': True,
        'USE_MULTI_STRATEGY_LINKING': True,
        'USE_ADAPTIVE_UMLS': True,
        'USE_UMLS_NORMALIZATION': True,
        'USE_REASONING_RULES': True,
        'USE_KG_GUIDED_REASONING': False,  # 关闭知识图谱引导推理
        'USE_OPTIMIZED_MULTIHOP': True,
        'USE_ENHANCED_ANSWER_GEN': True
    },
    'ablation_multihop': {
        'USE_HIERARCHICAL_KG': True,
        'USE_MULTI_STRATEGY_LINKING': True,
        'USE_ADAPTIVE_UMLS': True,
        'USE_UMLS_NORMALIZATION': True,
        'USE_REASONING_RULES': True,
        'USE_KG_GUIDED_REASONING': True,
        'USE_OPTIMIZED_MULTIHOP': False,   # 关闭优化多跳推理
        'USE_ENHANCED_ANSWER_GEN': True
    },
    'ablation_enhanced_answer': {
        'USE_HIERARCHICAL_KG': True,
        'USE_MULTI_STRATEGY_LINKING': True,
        'USE_ADAPTIVE_UMLS': True,
        'USE_UMLS_NORMALIZATION': True,
        'USE_REASONING_RULES': True,
        'USE_KG_GUIDED_REASONING': True,
        'USE_OPTIMIZED_MULTIHOP': True,
        'USE_ENHANCED_ANSWER_GEN': False   # 关闭增强答案生成
    },
    # 自定义配置 - 根据实际需要调整的配置
    'myself_settings': {
        'USE_HIERARCHICAL_KG': True,
        'USE_MULTI_STRATEGY_LINKING': True,
        'USE_ADAPTIVE_UMLS': True,
        'USE_UMLS_NORMALIZATION': True,
        'USE_REASONING_RULES': False,      # 自定义：关闭推理规则
        'USE_KG_GUIDED_REASONING': False,  # 自定义：关闭KG引导推理
        'USE_OPTIMIZED_MULTIHOP': True,
        'USE_ENHANCED_ANSWER_GEN': True
    }
}

# 当前实验配置 (可以通过命令行参数或环境变量修改)
# 从环境变量ABLATION_CONFIG读取配置名称，默认使用'myself_settings'
CURRENT_ABLATION_CONFIG = os.getenv('ABLATION_CONFIG', 'myself_settings')

# 配置日志系统
logging.basicConfig(level=logging.INFO)  # 设置日志级别为INFO
logger = logging.getLogger(__name__)      # 获取当前模块的日志记录器

def get_ablation_config():
    """
    获取当前消融实验配置
    返回当前配置的字典，如果配置名不存在则使用完整模型配置
    """
    # 根据配置名获取对应的配置字典
    config = ABLATION_CONFIGS.get(CURRENT_ABLATION_CONFIG, ABLATION_CONFIGS['full_model'])
    # 记录当前使用的配置信息
    logger.info(f"🔬 Using ablation configuration: {CURRENT_ABLATION_CONFIG}")
    logger.info(f"📋 Configuration details: {config}")
    return config

# 获取当前配置并存储为全局变量
ABLATION_CONFIG = get_ablation_config()

# ========================= 性能优化配置 =========================
# 系统性能相关参数
CLEANUP_FREQUENCY = 15      # 每处理15个样本后进行一次资源清理
MAX_CACHE_SIZE = 1500       # 缓存最大容量
KEEP_CACHE_SIZE = 800       # 清理后保留的缓存大小
MAX_FAILED_CUIS = 1000      # 最大失败CUI数量

# Enhanced API retry configuration
# API重试配置参数
MAX_RETRIES = 60                    # 最大重试次数
RETRY_WAIT_TIME = 60                # 重试等待时间(秒)
ENTITY_CONFIDENCE_THRESHOLD = 0.85   # 实体匹配置信度阈值
KNOWLEDGE_QUALITY_THRESHOLD = 0.7    # 知识质量阈值
MIN_SIMILARITY_THRESHOLD = 0.6       # 最小相似度阈值

# ========================= 层次化知识图谱架构 =========================

class HierarchicalKGFramework:
    """
    层次化知识图谱框架类
    用于构建和管理医学知识的层次化结构
    """
    def __init__(self):
        """初始化层次化知识图谱框架"""
        # 初始化各种层次化结构的字典
        self.disease_hierarchy = defaultdict(list)    # 疾病层次结构
        self.symptom_hierarchy = defaultdict(list)     # 症状层次结构
        self.treatment_hierarchy = defaultdict(list)   # 治疗层次结构
        self.anatomy_hierarchy = defaultdict(list)     # 解剖结构层次
        self.pathology_hierarchy = defaultdict(list)   # 病理层次结构
        
        # 定义不同关系类型的权重
        self.hierarchy_weights = {
            'is_a': 1.0,        # "是一个"关系权重最高
            'part_of': 0.9,     # "部分"关系
            'subtype_of': 0.95, # "子类型"关系
            'category_of': 0.8, # "类别"关系
            'related_to': 0.6   # "相关"关系权重最低
        }
    
    def build_hierarchical_structure(self, flat_kg):
        """
        构建层次化知识结构
        输入: flat_kg - 平坦的知识图谱三元组列表
        """
        # 检查是否启用层次化KG功能
        if not ABLATION_CONFIG['USE_HIERARCHICAL_KG']:
            logger.info("🔬 Hierarchical KG Framework disabled in ablation study")
            return
            
        logger.info("Building hierarchical knowledge structure...")
        
        # 构建各种层次结构
        self._build_disease_hierarchy(flat_kg)    # 构建疾病层次
        self._build_symptom_hierarchy(flat_kg)    # 构建症状层次
        self._build_treatment_hierarchy(flat_kg)  # 构建治疗层次
        self._build_anatomy_hierarchy(flat_kg)    # 构建解剖层次
        
        # 记录构建的层次结构统计信息
        logger.info(f"Built hierarchies: diseases={len(self.disease_hierarchy)}, "
                   f"symptoms={len(self.symptom_hierarchy)}, "
                   f"treatments={len(self.treatment_hierarchy)}")
    
    def _build_disease_hierarchy(self, flat_kg):
        """构建疾病分类层次"""
        # 遍历所有三元组
        for triple in flat_kg:
            if len(triple) >= 3:  # 确保三元组完整
                head, relation, tail = triple[0], triple[1], triple[2]
                
                # 检查是否是层次关系
                if any(keyword in relation.lower() for keyword in 
                       ['is_a', 'subtype', 'category', 'type_of']):
                    # 检查头实体是否为疾病类型
                    if any(keyword in head.lower() for keyword in 
                           ['disease', 'syndrome', 'disorder', 'condition']):
                        # 将层次信息添加到疾病层次字典中
                        self.disease_hierarchy[tail].append({
                            'entity': head,     # 实体名称
                            'relation': relation, # 关系类型
                            'weight': self.hierarchy_weights.get(relation.lower(), 0.5)  # 权重
                        })
    
    def _build_symptom_hierarchy(self, flat_kg):
        """构建症状-疾病关联层次"""
        for triple in flat_kg:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                # 检查是否是症状相关关系
                if any(keyword in relation.lower() for keyword in 
                       ['symptom', 'sign', 'manifestation', 'presents']):
                    # 构建症状层次结构
                    self.symptom_hierarchy[head].append({
                        'entity': tail,
                        'relation': relation,
                        'weight': self.hierarchy_weights.get(relation.lower(), 0.7)
                    })
    
    def _build_treatment_hierarchy(self, flat_kg):
        """构建治疗方案层次"""
        for triple in flat_kg:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                # 检查是否是治疗相关关系
                if any(keyword in relation.lower() for keyword in 
                       ['treat', 'therapy', 'medication', 'drug']):
                    # 构建治疗层次结构
                    self.treatment_hierarchy[head].append({
                        'entity': tail,
                        'relation': relation,
                        'weight': self.hierarchy_weights.get(relation.lower(), 0.8)
                    })
    
    def _build_anatomy_hierarchy(self, flat_kg):
        """构建解剖结构层次"""
        for triple in flat_kg:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                # 检查是否是解剖结构相关关系
                if any(keyword in relation.lower() for keyword in 
                       ['part_of', 'located_in', 'contains', 'anatomy']):
                    # 构建解剖层次结构
                    self.anatomy_hierarchy[tail].append({
                        'entity': head,
                        'relation': relation,
                        'weight': self.hierarchy_weights.get(relation.lower(), 0.6)
                    })
    
    def get_hierarchical_context(self, entity, context_type='all'):
        """
        获取实体的层次化上下文
        输入: entity - 实体名称, context_type - 上下文类型
        返回: 包含不同层次上下文的字典
        """
        # 检查是否启用层次化KG功能
        if not ABLATION_CONFIG['USE_HIERARCHICAL_KG']:
            return {}
            
        context = {}  # 初始化上下文字典
        
        # 根据请求的上下文类型添加相应信息
        if context_type in ['all', 'disease']:
            context['diseases'] = self.disease_hierarchy.get(entity, [])
        
        if context_type in ['all', 'symptom']:
            context['symptoms'] = self.symptom_hierarchy.get(entity, [])
        
        if context_type in ['all', 'treatment']:
            context['treatments'] = self.treatment_hierarchy.get(entity, [])
        
        if context_type in ['all', 'anatomy']:
            context['anatomy'] = self.anatomy_hierarchy.get(entity, [])
        
        return context

# ========================= 多策略实体链接 =========================

class SemanticMatcher:
    """语义匹配器类 - 基于语义相似度进行实体匹配"""
    def __init__(self):
        self.similarity_threshold = 0.7  # 相似度阈值
    
    def match(self, entities, umls_kg):
        """
        语义相似度匹配
        输入: entities - 待匹配实体列表, umls_kg - UMLS知识图谱
        返回: 匹配结果字典
        """
        # 检查是否启用多策略链接
        if not ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
            return {}
            
        matches = {}  # 存储匹配结果
        
        # 遍历每个实体进行匹配
        for entity in entities:
            best_match = None   # 最佳匹配
            best_score = 0      # 最佳分数
            
            # 在UMLS知识图谱中寻找最佳匹配
            for kg_entity in umls_kg:
                score = self._calculate_semantic_similarity(entity, kg_entity)
                if score > best_score and score > self.similarity_threshold:
                    best_score = score
                    best_match = kg_entity
            
            # 如果找到了匹配，记录结果
            if best_match:
                matches[entity] = {
                    'match': best_match, 
                    'score': best_score, 
                    'method': 'semantic'
                }
        
        return matches
    
    def _calculate_semantic_similarity(self, entity1, entity2):
        """
        计算语义相似度
        使用Jaccard相似度（交集/并集）计算两个实体的相似度
        """
        # 分词并转为集合
        words1 = set(entity1.lower().split())
        words2 = set(entity2.lower().split())
        
        # 计算交集和并集
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        # 返回Jaccard相似度
        return intersection / union if union > 0 else 0

class ContextAwareLinker:
    """上下文感知链接器类 - 基于上下文信息进行实体链接"""
    def __init__(self):
        self.context_weight = 0.3  # 上下文权重
    
    def link(self, entities, context):
        """
        上下文感知链接
        输入: entities - 实体列表, context - 上下文文本
        返回: 链接结果字典
        """
        # 检查是否启用多策略链接
        if not ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
            return {}
            
        links = {}  # 存储链接结果
        context_words = set(context.lower().split())  # 上下文词集合
        
        # 为每个实体计算上下文相关分数
        for entity in entities:
            entity_words = set(entity.lower().split())
            # 计算实体词与上下文词的重叠度
            context_overlap = len(entity_words.intersection(context_words))
            context_score = context_overlap / len(entity_words) if entity_words else 0
            
            links[entity] = {
                'context_score': context_score,
                'method': 'context_aware'
            }
        
        return links

class ConfidenceEstimator:
    """置信度估计器类 - 融合多种匹配策略的结果"""
    def __init__(self):
        self.weight_semantic = 0.6  # 语义权重
        self.weight_context = 0.4   # 上下文权重
    
    def fuse_results(self, semantic_matches, context_matches):
        """
        置信度估计和融合
        输入: semantic_matches - 语义匹配结果, context_matches - 上下文匹配结果
        返回: 融合后的链接结果
        """
        # 检查是否启用多策略链接
        if not ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
            return {}
            
        final_links = {}  # 最终链接结果
        
        # 获取所有实体的并集
        all_entities = set(semantic_matches.keys()) | set(context_matches.keys())
        
        # 为每个实体计算综合分数
        for entity in all_entities:
            # 获取语义分数和上下文分数
            semantic_score = semantic_matches.get(entity, {}).get('score', 0)
            context_score = context_matches.get(entity, {}).get('context_score', 0)
            
            # 计算加权综合分数
            combined_score = (self.weight_semantic * semantic_score + 
                            self.weight_context * context_score)
            
            # 存储融合结果
            final_links[entity] = {
                'final_score': combined_score,
                'semantic_score': semantic_score,
                'context_score': context_score,
                'method': 'fused'
            }
        
        return final_links

class EnhancedEntityLinking:
    """增强实体链接类 - 集成多种链接策略"""
    def __init__(self):
        # 初始化各个组件
        self.semantic_matcher = SemanticMatcher()           # 语义匹配器
        self.context_aware_linker = ContextAwareLinker()    # 上下文感知链接器
        self.confidence_estimator = ConfidenceEstimator()   # 置信度估计器
    
    def multi_strategy_linking(self, entities, context, umls_kg):
        """
        多策略实体链接
        输入: entities - 实体列表, context - 上下文, umls_kg - UMLS知识图谱
        返回: 最终的链接结果
        """
        # 检查是否启用多策略链接
        if not ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
            return {}
            
        # 执行语义匹配
        semantic_matches = self.semantic_matcher.match(entities, umls_kg)
        # 执行上下文链接
        context_matches = self.context_aware_linker.link(entities, context)
        # 融合结果
        final_links = self.confidence_estimator.fuse_results(
            semantic_matches, context_matches
        )
        
        return final_links

# ========================= 自适应UMLS知识选择 =========================

class AdaptiveUMLSSelector:
    """自适应UMLS选择器类 - 根据问题类型选择相关的UMLS知识"""
    def __init__(self, umls_api):
        self.umls_api = umls_api  # UMLS API接口
        
        # 针对不同任务类型的语义类型权重
        self.task_specific_weights = {
            'treatment': {  # 治疗相关任务
                'therapeutic_procedure': 2.0,     # 治疗程序
                'pharmacologic_substance': 1.8,   # 药理物质
                'clinical_drug': 1.6              # 临床药物
            },
            'diagnosis': {  # 诊断相关任务
                'disease_syndrome': 2.0,    # 疾病综合征
                'sign_symptom': 1.8,        # 症状体征
                'finding': 1.6              # 发现
            },
            'causation': {  # 因果关系任务
                'disease_syndrome': 1.8,      # 疾病综合征
                'pathologic_function': 1.6,  # 病理功能
                'injury_poisoning': 1.4       # 伤害中毒
            },
            'prevention': {  # 预防相关任务
                'therapeutic_procedure': 1.8,   # 治疗程序
                'preventive_procedure': 2.0,    # 预防程序
                'health_care_activity': 1.6     # 医疗保健活动
            }
        }
    
    def select_relevant_umls_knowledge(self, question_type, entities):
        """
        根据问题类型选择相关UMLS知识
        输入: question_type - 问题类型, entities - 实体列表
        返回: 相关的UMLS知识
        """
        # 检查是否启用自适应UMLS
        if not ABLATION_CONFIG['USE_ADAPTIVE_UMLS']:
            return self.get_general_knowledge(entities)
            
        # 根据问题类型调用相应的知识获取方法
        if question_type == 'treatment':
            return self.get_treatment_focused_knowledge(entities)
        elif question_type == 'diagnosis':
            return self.get_diagnosis_focused_knowledge(entities)
        elif question_type == 'causation':
            return self.get_causation_focused_knowledge(entities)
        elif question_type == 'prevention':
            return self.get_prevention_focused_knowledge(entities)
        else:
            return self.get_general_knowledge(entities)
    
    def get_treatment_focused_knowledge(self, entities):
        """获取治疗相关的知识"""
        treatment_knowledge = []  # 存储治疗知识
        
        # 遍历每个实体
        for entity in entities:
            # 搜索UMLS概念
            concepts = self.umls_api.search_concepts(entity)
            if concepts and 'results' in concepts:
                # 处理前5个概念
                for concept in concepts['results'][:5]:
                    cui = concept.get('ui', '')  # 获取CUI
                    relations = self.umls_api.get_concept_relations(cui)  # 获取关系
                    
                    # 筛选治疗相关关系
                    treatment_relations = [
                        rel for rel in relations 
                        if any(keyword in rel.get('relationLabel', '').lower() 
                               for keyword in ['treat', 'therapy', 'medication'])
                    ]
                    
                    if treatment_relations:
                        treatment_knowledge.extend(treatment_relations)
        
        return treatment_knowledge
    
    def get_diagnosis_focused_knowledge(self, entities):
        """获取诊断相关的知识"""
        diagnosis_knowledge = []  # 存储诊断知识
        
        for entity in entities:
            concepts = self.umls_api.search_concepts(entity)
            if concepts and 'results' in concepts:
                for concept in concepts['results'][:5]:
                    cui = concept.get('ui', '')
                    relations = self.umls_api.get_concept_relations(cui)
                    
                    # 筛选诊断相关关系
                    diagnosis_relations = [
                        rel for rel in relations 
                        if any(keyword in rel.get('relationLabel', '').lower() 
                               for keyword in ['diagnose', 'symptom', 'sign', 'finding'])
                    ]
                    
                    if diagnosis_relations:
                        diagnosis_knowledge.extend(diagnosis_relations)
        
        return diagnosis_knowledge
    
    def get_causation_focused_knowledge(self, entities):
        """获取因果关系相关的知识"""
        causation_knowledge = []  # 存储因果知识
        
        for entity in entities:
            concepts = self.umls_api.search_concepts(entity)
            if concepts and 'results' in concepts:
                for concept in concepts['results'][:5]:
                    cui = concept.get('ui', '')
                    relations = self.umls_api.get_concept_relations(cui)
                    
                    # 筛选因果关系
                    causation_relations = [
                        rel for rel in relations 
                        if any(keyword in rel.get('relationLabel', '').lower() 
                               for keyword in ['cause', 'lead', 'result', 'induce'])
                    ]
                    
                    if causation_relations:
                        causation_knowledge.extend(causation_relations)
        
        return causation_knowledge
    
    def get_prevention_focused_knowledge(self, entities):
        """获取预防相关的知识"""
        prevention_knowledge = []  # 存储预防知识
        
        for entity in entities:
            concepts = self.umls_api.search_concepts(entity)
            if concepts and 'results' in concepts:
                for concept in concepts['results'][:5]:
                    cui = concept.get('ui', '')
                    relations = self.umls_api.get_concept_relations(cui)
                    
                    # 筛选预防相关关系
                    prevention_relations = [
                        rel for rel in relations 
                        if any(keyword in rel.get('relationLabel', '').lower() 
                               for keyword in ['prevent', 'avoid', 'reduce_risk'])
                    ]
                    
                    if prevention_relations:
                        prevention_knowledge.extend(prevention_relations)
        
        return prevention_knowledge
    
    def get_general_knowledge(self, entities):
        """获取通用知识"""
        general_knowledge = []  # 存储通用知识
        
        for entity in entities:
            concepts = self.umls_api.search_concepts(entity)
            if concepts and 'results' in concepts:
                # 对于通用知识，获取前3个概念的前10个关系
                for concept in concepts['results'][:3]:
                    cui = concept.get('ui', '')
                    relations = self.umls_api.get_concept_relations(cui)
                    general_knowledge.extend(relations[:10])
        
        return general_knowledge

# ========================= 知识图谱引导的思维链推理 =========================

class SchemaReasoner:
    """模式推理器类 - 基于医学模式进行推理"""
    def __init__(self):
        # 定义不同问题类型对应的医学推理模式
        self.medical_schemas = {
            'diagnosis': ['symptom', 'finding', 'test', 'disease'],      # 诊断模式
            'treatment': ['disease', 'medication', 'procedure', 'outcome'], # 治疗模式
            'causation': ['risk_factor', 'cause', 'disease', 'complication'], # 因果模式
            'prevention': ['risk_factor', 'intervention', 'prevention', 'outcome'] # 预防模式
        }
    
    def infer_paths(self, question, kg):
        """
        基于模式推理路径
        输入: question - 问题文本, kg - 知识图谱
        返回: 推理路径列表
        """
        # 检查是否启用KG引导推理
        if not ABLATION_CONFIG['USE_KG_GUIDED_REASONING']:
            return []
            
        # 识别问题模式
        question_type = self._identify_question_schema(question)
        schema = self.medical_schemas.get(question_type, [])
        
        reasoning_paths = []  # 存储推理路径
        # 在模式中寻找连续的概念对
        for i in range(len(schema) - 1):
            start_type = schema[i]      # 起始概念类型
            end_type = schema[i + 1]    # 终止概念类型
            # 寻找符合模式的路径
            paths = self._find_schema_paths(kg, start_type, end_type)
            reasoning_paths.extend(paths)
        
        return reasoning_paths
    
    def _identify_question_schema(self, question):
        """
        识别问题模式
        输入: question - 问题文本
        返回: 问题类型字符串
        """
        question_lower = question.lower()
        
        # 根据关键词识别问题类型
        if any(keyword in question_lower for keyword in ['treat', 'therapy', 'medication']):
            return 'treatment'
        elif any(keyword in question_lower for keyword in ['cause', 'why', 'due to']):
            return 'causation'
        elif any(keyword in question_lower for keyword in ['prevent', 'avoid', 'reduce risk']):
            return 'prevention'
        else:
            return 'diagnosis'  # 默认为诊断类型
    
    def _find_schema_paths(self, kg, start_type, end_type):
        """
        查找符合模式的路径
        输入: kg - 知识图谱, start_type - 起始类型, end_type - 结束类型
        返回: 符合条件的路径列表
        """
        paths = []  # 存储找到的路径
        
        # 遍历知识图谱中的三元组
        for triple in kg:
            if len(triple) >= 3:
                # 检查三元组是否符合模式
                if (start_type in triple[0].lower() or start_type in triple[1].lower()) and \
                   (end_type in triple[2].lower() or end_type in triple[1].lower()):
                    paths.append(triple)
        return paths

class KGGuidedReasoningEngine:
    """知识图谱引导推理引擎类"""
    def __init__(self, kg, llm):
        self.kg = kg                              # 知识图谱
        self.llm = llm                           # 语言模型
        self.schema_reasoner = SchemaReasoner()  # 模式推理器
    
    def kg_guided_reasoning(self, question, kg_subgraph):
        """
        知识图谱引导的推理
        输入: question - 问题, kg_subgraph - 知识图谱子图
        返回: 推理结果
        """
        # 检查是否启用KG引导推理
        if not ABLATION_CONFIG['USE_KG_GUIDED_REASONING']:
            return "KG-guided reasoning disabled in ablation study"
            
        # 获取模式路径
        schema_paths = self.schema_reasoner.infer_paths(question, self.kg)
        # 生成最优子图
        optimal_subgraph = self.generate_optimal_subgraph(
            question, schema_paths, kg_subgraph
        )
        # 使用LLM进行推理
        reasoning_result = self.llm_reasoning_with_kg(question, optimal_subgraph)
        
        return reasoning_result
    
    def generate_optimal_subgraph(self, question, schema_paths, kg_subgraph):
        """
        生成最优子图
        输入: question - 问题, schema_paths - 模式路径, kg_subgraph - 原始子图
        返回: 优化后的子图
        """
        # 合并模式路径和原始子图
        combined_graph = kg_subgraph + schema_paths
        
        scored_triples = []  # 存储评分后的三元组
        # 为每个三元组计算相关性分数
        for triple in combined_graph:
            score = self._calculate_relevance_score(question, triple)
            scored_triples.append((triple, score))
        
        # 按分数降序排列
        scored_triples.sort(key=lambda x: x[1], reverse=True)
        # 选择前15个最相关的三元组
        optimal_subgraph = [triple for triple, score in scored_triples[:15]]
        
        return optimal_subgraph
    
    def _calculate_relevance_score(self, question, triple):
        """
        计算三元组与问题的相关性分数
        输入: question - 问题, triple - 三元组
        返回: 相关性分数
        """
        question_words = set(question.lower().split())  # 问题词集合
        triple_words = set()                            # 三元组词集合
        
        # 提取三元组中的所有词
        for element in triple:
            triple_words.update(element.lower().split())
        
        # 计算词汇重叠度
        overlap = len(question_words.intersection(triple_words))
        relevance_score = overlap / len(question_words) if question_words else 0
        
        return relevance_score
    
    def llm_reasoning_with_kg(self, question, kg_subgraph):
        """
        使用LLM进行知识图谱增强推理
        输入: question - 问题, kg_subgraph - 知识图谱子图
        返回: 推理结果
        """
        # 将知识图谱格式化为文本
        kg_context = "\n".join([f"{t[0]} -> {t[1]} -> {t[2]}" for t in kg_subgraph])
        
        # 构建推理提示
        prompt = f"""
        Question: {question}
        
        Knowledge Graph Context:
        {kg_context}
        
        Based on the structured medical knowledge above, provide step-by-step reasoning to answer the question.
        Focus on the relationships and pathways shown in the knowledge graph.
        """
        
        try:
            # 调用LLM进行推理
            response = self.llm([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Error in LLM reasoning: {e}")
            return "Unable to generate reasoning based on knowledge graph."

# ========================= 优化多跳推理 =========================

class PathRanker:
    """路径排名器类 - 基于医学知识对路径进行排序"""
    def __init__(self):
        # 定义医学关系的权重
        self.medical_relation_weights = {
            'causes': 3.0,         # 因果关系权重最高
            'treats': 2.8,         # 治疗关系
            'prevents': 2.5,       # 预防关系
            'symptom_of': 2.2,     # 症状关系
            'diagnoses': 2.0,      # 诊断关系
            'associated_with': 1.8, # 关联关系
            'located_in': 1.5,     # 位置关系
            'part_of': 1.2,        # 部分关系
            'related_to': 1.0      # 相关关系权重最低
        }
    
    def rank_by_quality(self, paths):
        """
        根据质量对路径进行排序
        输入: paths - 路径列表
        返回: 按质量排序的路径列表
        """
        # 检查是否启用优化多跳推理
        if not ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP']:
            return paths
            
        scored_paths = []  # 存储评分后的路径
        
        # 为每个路径计算质量分数
        for path in paths:
            quality_score = self._calculate_path_quality(path)
            scored_paths.append((path, quality_score))
        
        # 按分数降序排列
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        return [path for path, score in scored_paths]
    
    def _calculate_path_quality(self, path):
        """
        计算路径质量分数
        输入: path - 路径
        返回: 质量分数
        """
        if not path:
            return 0
        
        relation_score = 0  # 关系分数
        # 遍历路径中的每一步
        for step in path:
            if len(step) >= 2:
                relation = step[1].lower()  # 关系名称
                # 查找匹配的关系权重
                for key, weight in self.medical_relation_weights.items():
                    if key in relation:
                        relation_score += weight
                        break
                else:
                    # 如果没有找到匹配的关系，给予默认权重
                    relation_score += 0.5
        
        # 路径长度惩罚
        length_penalty = len(path) * 0.1
        quality_score = relation_score - length_penalty
        
        return quality_score

class OptimizedMultiHopReasoning:
    """优化多跳推理类"""
    def __init__(self, kg, path_ranker=None):
        self.kg = kg                                          # 知识图谱
        self.path_ranker = path_ranker or PathRanker()       # 路径排名器
        self.reasoning_cache = {}                            # 推理缓存
    
    def intelligent_path_selection(self, start_entities, target_entities, max_hops=3):
        """
        智能路径选择
        输入: start_entities - 起始实体列表, target_entities - 目标实体列表, max_hops - 最大跳数
        返回: 选择的路径列表
        """
        # 检查是否启用优化多跳推理
        if not ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP']:
            return self._basic_path_selection(start_entities, target_entities, max_hops)
            
        # 计算医学相关性权重
        weighted_paths = self.calculate_medical_relevance_weights(
            start_entities, target_entities
        )
        
        # 动态剪枝
        pruned_paths = self.dynamic_pruning(weighted_paths, max_hops)
        # 质量排序
        quality_ranked_paths = self.path_ranker.rank_by_quality(pruned_paths)
        
        return quality_ranked_paths
    
    def _basic_path_selection(self, start_entities, target_entities, max_hops):
        """基础版本的路径选择（用于消融实验）"""
        basic_paths = []
        # 简单地寻找连接路径
        for start_entity in start_entities:
            for target_entity in target_entities:
                paths = self._find_connecting_paths(start_entity, target_entity)
                basic_paths.extend(paths[:3])  # 只取前3个路径
        return basic_paths
    
    def calculate_medical_relevance_weights(self, start_entities, target_entities):
        """
        计算基于医学知识的路径权重
        输入: start_entities - 起始实体, target_entities - 目标实体
        返回: 带权重的路径列表
        """
        weighted_paths = []  # 存储带权重的路径
        
        # 遍历起始实体和目标实体的组合
        for start_entity in start_entities:
            for target_entity in target_entities:
                cache_key = f"{start_entity}-{target_entity}"
                
                # 检查缓存
                if cache_key in self.reasoning_cache:
                    weighted_paths.extend(self.reasoning_cache[cache_key])
                    continue
                
                # 寻找连接路径
                paths = self._find_connecting_paths(start_entity, target_entity)
                
                # 为每个路径计算医学相关性权重
                for path in paths:
                    weight = self._calculate_medical_relevance(path)
                    weighted_paths.append((path, weight))
                
                # 缓存结果
                self.reasoning_cache[cache_key] = [(path, weight) for path, weight in weighted_paths[-len(paths):]]
        
        return weighted_paths
    
    def dynamic_pruning(self, weighted_paths, max_hops):
        """
        动态剪枝策略
        输入: weighted_paths - 带权重的路径, max_hops - 最大跳数
        返回: 剪枝后的路径列表
        """
        pruned_paths = []  # 存储剪枝后的路径
        
        # 按权重降序排列
        weighted_paths.sort(key=lambda x: x[1], reverse=True)
        
        # 应用剪枝策略
        for path, weight in weighted_paths:
            # 长度限制
            if len(path) <= max_hops:
                # 权重阈值
                if weight > 0.5:
                    pruned_paths.append(path)
            
            # 数量限制
            if len(pruned_paths) >= 20:
                break
        
        return pruned_paths
    
    def _find_connecting_paths(self, start_entity, target_entity):
        """
        查找连接路径
        输入: start_entity - 起始实体, target_entity - 目标实体
        返回: 连接路径列表
        """
        paths = []  # 存储找到的路径
        
        # 查找直接连接
        for triple in self.kg:
            if len(triple) >= 3:
                if triple[0] == start_entity and triple[2] == target_entity:
                    paths.append([triple])  # 单跳路径
        
        # 查找两跳路径
        intermediate_entities = set()  # 中间实体集合
        # 找出所有从起始实体出发的实体
        for triple in self.kg:
            if len(triple) >= 3 and triple[0] == start_entity:
                intermediate_entities.add(triple[2])
        
        # 通过中间实体连接到目标实体
        for intermediate in intermediate_entities:
            for triple in self.kg:
                if len(triple) >= 3 and triple[0] == intermediate and triple[2] == target_entity:
                    # 找到第一跳的三元组
                    first_hop = next((t for t in self.kg if len(t) >= 3 and t[0] == start_entity and t[2] == intermediate), None)
                    if first_hop:
                        paths.append([first_hop, triple])  # 两跳路径
        
        return paths[:10]  # 限制返回的路径数量
    
    def _calculate_medical_relevance(self, path):
        """
        计算医学相关性
        输入: path - 路径
        返回: 相关性分数
        """
        relevance_score = 0  # 相关性分数
        
        # 遍历路径中的每一步
        for step in path:
            if len(step) >= 3:
                # 计算实体分数和关系分数
                entity_score = self._get_entity_medical_score(step[0]) + self._get_entity_medical_score(step[2])
                relation_score = self._get_relation_medical_score(step[1])
                relevance_score += entity_score + relation_score
        
        # 返回平均相关性分数
        return relevance_score / len(path) if path else 0
    
    def _get_entity_medical_score(self, entity):
        """
        获取实体的医学相关性分数
        输入: entity - 实体名称
        返回: 医学相关性分数
        """
        medical_keywords = ['disease', 'symptom', 'treatment', 'medication', 'diagnosis', 'therapy']
        entity_lower = entity.lower()
        
        score = 0
        # 检查实体是否包含医学关键词
        for keyword in medical_keywords:
            if keyword in entity_lower:
                score += 1
        
        return score
    
    def _get_relation_medical_score(self, relation):
        """
        获取关系的医学相关性分数
        输入: relation - 关系名称
        返回: 医学相关性分数
        """
        relation_weights = {
            'causes': 3.0, 'treats': 2.8, 'prevents': 2.5,
            'symptom_of': 2.2, 'diagnoses': 2.0, 'associated_with': 1.8
        }
        
        relation_lower = relation.lower()
        # 查找匹配的关系权重
        for key, weight in relation_weights.items():
            if key in relation_lower:
                return weight
        
        return 1.0  # 默认权重

# ========================= UMLS API集成 =========================

class UMLS_API:
    """UMLS API客户端类 - 用于访问UMLS统一医学语言系统"""
    def __init__(self, api_key, version="current"):
        """
        初始化UMLS API客户端
        输入: api_key - API密钥, version - API版本
        """
        self.api_key = api_key    # API密钥
        self.version = version    # API版本
        # 构建API端点URL
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        
        # 初始化HTTP会话
        self.session = requests.Session()
        self.session.timeout = 30  # 设置超时时间
        
        # 初始化缓存
        self.cache = {}              # API响应缓存
        self.cache_size = 10000      # 缓存大小限制
        self.failed_cuis = set()     # 失败的CUI集合
        
        # 检查是否需要测试连接
        if ABLATION_CONFIG['USE_UMLS_NORMALIZATION'] or ABLATION_CONFIG['USE_ADAPTIVE_UMLS']:
            try:
                self._test_connection()  # 测试API连接
                logger.info("UMLS API connection successful")
            except Exception as e:
                logger.warning(f"UMLS API connection failed: {e}. Operating in offline mode.")
        else:
            logger.info("🔬 UMLS API disabled in ablation study")
    
    def _test_connection(self):
        """测试API连接"""
        try:
            # 构建测试请求参数
            params = {
                "string": "pain",        # 测试搜索词
                "apiKey": self.api_key,  # API密钥
                "pageNumber": 1,         # 页码
                "pageSize": 1            # 页面大小
            }
            # 发送测试请求
            response = self.session.get(self.search_url, params=params, timeout=5)
            response.raise_for_status()  # 检查HTTP错误
            data = response.json()       # 解析JSON响应
            # 检查响应格式
            if 'result' not in data:
                raise Exception("Invalid API response format")
        except Exception as e:
            raise Exception(f"API connection test failed: {e}")
    
    def search_concepts(self, search_string, search_type="words", page_size=25):
        """
        搜索UMLS概念
        输入: search_string - 搜索字符串, search_type - 搜索类型, page_size - 页面大小
        返回: 搜索结果字典
        """
        # 检查是否启用UMLS功能
        if not (ABLATION_CONFIG['USE_UMLS_NORMALIZATION'] or ABLATION_CONFIG['USE_ADAPTIVE_UMLS']):
            return None
            
        # 构建缓存键
        cache_key = f"search_{search_string}_{page_size}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # 构建请求参数
            params = {
                "string": search_string,
                "apiKey": self.api_key,
                "pageNumber": 1,
                "pageSize": page_size
            }
            
            # 发送搜索请求
            response = self.session.get(self.search_url, params=params)
            response.raise_for_status()     # 检查HTTP错误
            response.encoding = 'utf-8'     # 设置编码
            
            data = response.json()          # 解析响应
            result = data.get("result", {}) # 提取结果
            
            # 缓存结果
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching concepts for '{search_string}': {e}")
            return None
    
    def get_concept_details(self, cui):
        """
        获取概念详细信息
        输入: cui - 概念唯一标识符
        返回: 概念详细信息字典
        """
        # 检查是否启用UMLS功能
        if not (ABLATION_CONFIG['USE_UMLS_NORMALIZATION'] or ABLATION_CONFIG['USE_ADAPTIVE_UMLS']):
            return None
            
        # 检查缓存
        cache_key = f"details_{cui}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # 构建请求URL
            url = f"{self.content_url}/CUI/{cui}"
            params = {"apiKey": self.api_key}
            
            # 发送请求
            response = self.session.get(url, params=params)
            response.raise_for_status()
            response.encoding = "utf-8"
            
            # 解析响应
            data = response.json()
            result = data.get("result", {})
            
            # 缓存结果
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting details for CUI {cui}: {e}")
            return None
    
    def get_concept_atoms(self, cui):
        """
        获取概念的原子信息
        输入: cui - 概念唯一标识符
        返回: 原子信息列表
        """
        # 检查是否启用UMLS功能
        if not (ABLATION_CONFIG['USE_UMLS_NORMALIZATION'] or ABLATION_CONFIG['USE_ADAPTIVE_UMLS']):
            return None
            
        # 检查缓存
        cache_key = f"atoms_{cui}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # 构建请求URL
            url = f"{self.content_url}/CUI/{cui}/atoms"
            params = {"apiKey": self.api_key, "pageSize": 100}
            
            # 发送请求
            response = self.session.get(url, params=params)
            response.raise_for_status()
            response.encoding = "utf-8"
            
            # 解析响应
            data = response.json()
            result = data.get("result", [])
            
            # 缓存结果
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting atoms for CUI {cui}: {e}")
            return None
    
    def get_concept_relations(self, cui):
        """
        获取概念关系
        输入: cui - 概念唯一标识符
        返回: 关系列表
        """
        # 检查是否启用UMLS功能
        if not (ABLATION_CONFIG['USE_UMLS_NORMALIZATION'] or ABLATION_CONFIG['USE_ADAPTIVE_UMLS']):
            return []
            
        # 检查缓存
        cache_key = f"relations_{cui}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 检查是否在失败列表中
        if cui in self.failed_cuis:
            return []
        
        all_relations = []  # 存储所有关系
        
        try:
            # 分页获取关系信息
            for page in range(1, 6):  # 最多获取5页
                url = f"{self.content_url}/CUI/{cui}/relations"
                params = {
                    "apiKey": self.api_key,
                    "pageNumber": page,
                    "pageSize": 100
                }
                
                # 发送请求
                response = self.session.get(url, params=params)
                response.raise_for_status()
                response.encoding = "utf-8"
                
                # 解析响应
                data = response.json()
                page_relations = data.get("result", [])
                
                # 如果没有更多关系，跳出循环
                if not page_relations:
                    break
                    
                all_relations.extend(page_relations)
            
            # 缓存结果
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = all_relations
            
            return all_relations
            
        except requests.exceptions.HTTPError as e:
            if "404" in str(e):
                # 404错误表示CUI不存在，添加到失败缓存
                self.failed_cuis.add(cui)
                logger.warning(f"CUI {cui} not found (404), adding to failed cache")
            else:
                logger.error(f"HTTP error getting relations for CUI {cui}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting relations for CUI {cui}: {e}")
            return []

class UMLSNormalizer:
    """UMLS标准化器类 - 将医学术语标准化为UMLS概念"""
    def __init__(self, api_key):
        """
        初始化UMLS标准化器
        输入: api_key - UMLS API密钥
        """
        self.umls_api = UMLS_API(api_key)         # UMLS API客户端
        self.local_cache = {}                      # 本地缓存
        self.semantic_type_cache = {}              # 语义类型缓存
        
        # 初始化其他组件
        self.hierarchical_kg = HierarchicalKGFramework()
        self.enhanced_entity_linking = EnhancedEntityLinking()
        self.adaptive_umls_selector = AdaptiveUMLSSelector(self.umls_api)
        
        # UMLS语义类型优先级（T代码对应不同的医学概念类型）
        self.semantic_type_priority = {
            'T047': 10,  # Disease or Syndrome - 疾病或综合征
            'T184': 9,   # Sign or Symptom - 症状或体征
            'T061': 8,   # Therapeutic or Preventive Procedure - 治疗或预防程序
            'T121': 7,   # Pharmacologic Substance - 药理物质
            'T023': 6,   # Body Part, Organ, or Organ Component - 身体部位、器官或器官组件
            'T037': 5,   # Injury or Poisoning - 伤害或中毒
            'T046': 4,   # Pathologic Function - 病理功能
            'T033': 3,   # Finding - 发现
            'T170': 2,   # Intellectual Product - 智力产品
            'T169': 1    # Functional Concept - 功能概念
        }
    
    def _get_best_cui_for_term(self, term):
        """
        为给定术语获取最佳CUI（概念唯一标识符）
        输入: term - 医学术语
        返回: 最佳匹配的CUI
        """
        # 检查是否启用UMLS标准化
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return None
            
        # 检查本地缓存
        if term in self.local_cache:
            return self.local_cache[term]
        
        try:
            # 搜索UMLS概念
            search_results = self.umls_api.search_concepts(term)
            
            if not search_results or 'results' not in search_results:
                return None
            
            results = search_results['results']
            if not results:
                return None
            
            best_cui = None    # 最佳CUI
            best_score = 0     # 最佳分数
            
            # 遍历搜索结果，找到最佳匹配
            for result in results:
                cui = result['ui']     # CUI
                name = result['name']  # 概念名称
                
                # 计算匹配分数
                score = self._calculate_match_score(term, name, result)
                
                if score > best_score:
                    best_score = score
                    best_cui = cui
            
            # 缓存结果
            self.local_cache[term] = best_cui
            return best_cui
            
        except Exception as e:
            logger.error(f"Error getting CUI for term '{term}': {e}")
            return None
    
    def _calculate_match_score(self, original_term, concept_name, result):
        """
        计算匹配分数
        输入: original_term - 原始术语, concept_name - 概念名称, result - 搜索结果
        返回: 匹配分数
        """
        score = 0
        
        # 精确匹配得分最高
        if original_term.lower() == concept_name.lower():
            score += 100
        # 包含关系得分
        elif original_term.lower() in concept_name.lower():
            score += 50
        elif concept_name.lower() in original_term.lower():
            score += 30
        
        # 词汇重叠度
        original_words = set(original_term.lower().split())
        concept_words = set(concept_name.lower().split())
        overlap = len(original_words & concept_words)
        score += overlap * 10
        
        # 词根匹配
        if self._has_root_match(original_term, concept_name):
            score += 20
        
        return score
    
    def _has_root_match(self, term1, term2):
        """
        检查词根匹配
        输入: term1, term2 - 两个术语
        返回: 是否有词根匹配
        """
        # 定义常见后缀
        suffixes = ['s', 'es', 'ing', 'ed', 'er', 'est', 'ly']
        
        def get_root(word):
            """获取词根"""
            for suffix in suffixes:
                if word.endswith(suffix):
                    return word[:-len(suffix)]
            return word
        
        # 获取两个术语的词根
        root1 = get_root(term1.lower())
        root2 = get_root(term2.lower())
        
        # 检查词根是否匹配
        return root1 == root2 or root1 in root2 or root2 in root1
    
    def get_concept_synonyms(self, cui):
        """
        获取概念的同义词
        输入: cui - 概念唯一标识符
        返回: 同义词列表
        """
        # 检查是否启用UMLS标准化
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return []
            
        try:
            # 获取概念的原子信息
            atoms_result = self.umls_api.get_concept_atoms(cui)
            
            if not atoms_result:
                return []
            
            synonyms = []
            # 提取所有同义词
            for atom in atoms_result:
                name = atom.get('name', '')
                if name and name not in synonyms:
                    synonyms.append(name)
            
            return synonyms
            
        except Exception as e:
            logger.error(f"Error getting synonyms for CUI {cui}: {e}")
            return []
    
    def get_concept_relations(self, cui):
        """
        获取概念关系
        输入: cui - 概念唯一标识符
        返回: 关系列表
        """
        # 检查是否启用UMLS标准化
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return []
            
        try:
            # 获取概念关系
            relations_result = self.umls_api.get_concept_relations(cui)
            
            if not relations_result:
                return []
            
            relations = []
            # 处理关系信息
            for relation in relations_result:
                rel_type = relation.get('relationLabel', '')      # 关系类型
                related_cui = relation.get('relatedId', '')       # 相关CUI
                related_name = relation.get('relatedIdName', '')  # 相关概念名称
                
                if rel_type and related_cui:
                    relations.append({
                        'relation_type': rel_type,
                        'related_cui': related_cui,
                        'related_name': related_name
                    })
            
            return relations
            
        except Exception as e:
            logger.error(f"Error getting relations for CUI {cui}: {e}")
            return []
    
    def normalize_medical_terms(self, entities):
        """
        将医学术语标准化为UMLS概念
        输入: entities - 实体列表
        返回: 标准化后的实体列表
        """
        # 检查是否启用UMLS标准化
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return entities
            
        normalized_entities = []  # 存储标准化后的实体
        
        # 遍历每个实体进行标准化
        for entity in entities:
            try:
                # 获取最佳CUI
                cui = self._get_best_cui_for_term(entity)
                
                if cui:
                    # 获取概念详细信息
                    concept_details = self.umls_api.get_concept_details(cui)
                    
                    if concept_details:
                        # 使用首选名称
                        preferred_name = concept_details.get('name', entity)
                        normalized_entities.append(preferred_name)
                        logger.debug(f"标准化: {entity} -> {preferred_name} (CUI: {cui})")
                    else:
                        # 如果获取不到详细信息，保持原名称
                        normalized_entities.append(entity)
                else:
                    # 如果找不到CUI，保持原名称
                    normalized_entities.append(entity)
                    
            except Exception as e:
                logger.error(f"Error normalizing entity '{entity}': {e}")
                normalized_entities.append(entity)
        
        return normalized_entities
    
    def get_semantic_variants(self, entity):
        """
        获取实体的语义变体
        输入: entity - 实体名称
        返回: 语义变体列表
        """
        # 检查是否启用UMLS标准化
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return [entity]
            
        try:
            # 获取最佳CUI
            cui = self._get_best_cui_for_term(entity)
            if not cui:
                return [entity]
            
            # 获取同义词
            synonyms = self.get_concept_synonyms(cui)
            # 获取相关术语
            relations = self.get_concept_relations(cui)
            related_terms = []
            
            # 提取相关术语
            for relation in relations:
                if relation['relation_type'] in ['SY', 'PT', 'equivalent_to']:
                    related_terms.append(relation['related_name'])
            
            # 合并所有变体
            variants = [entity] + synonyms + related_terms
            
            # 去重并过滤
            unique_variants = []
            seen = set()
            
            for variant in variants:
                if variant and variant.lower() not in seen and len(variant) > 2:
                    seen.add(variant.lower())
                    unique_variants.append(variant)
            
            return unique_variants[:10]  # 返回前10个变体
            
        except Exception as e:
            logger.error(f"Error getting semantic variants for '{entity}': {e}")
            return [entity]
    
    def get_concept_hierarchy(self, entity):
        """
        获取概念层次结构
        输入: entity - 实体名称
        返回: 层次结构字典
        """
        # 检查是否启用UMLS标准化
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return {}
            
        try:
            # 获取最佳CUI
            cui = self._get_best_cui_for_term(entity)
            if not cui:
                return {}
            
            # 获取关系
            relations = self.get_concept_relations(cui)
            hierarchy = {
                'broader': [],   # 更广泛的概念
                'narrower': [],  # 更具体的概念
                'related': []    # 相关概念
            }
            
            # 分类关系
            for relation in relations:
                rel_type = relation['relation_type']
                related_name = relation['related_name']
                
                if rel_type in ['RB', 'inverse_isa', 'parent']:
                    hierarchy['broader'].append(related_name)
                elif rel_type in ['RN', 'isa', 'child']:
                    hierarchy['narrower'].append(related_name)
                elif rel_type in ['RT', 'related_to']:
                    hierarchy['related'].append(related_name)
            
            return hierarchy
            
        except Exception as e:
            logger.error(f"Error getting concept hierarchy for '{entity}': {e}")
            return {}
    
    def enhanced_entity_linking_method(self, entities, context, question_types):
        """
        增强的实体链接方法
        输入: entities - 实体列表, context - 上下文, question_types - 问题类型
        返回: 链接结果
        """
        # 检查是否启用多策略链接
        if not ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
            return {}
            
        try:
            # 构建UMLS知识图谱
            umls_kg = []
            for entity in entities:
                concepts = self.umls_api.search_concepts(entity)
                if concepts and 'results' in concepts:
                    umls_kg.extend([concept['name'] for concept in concepts['results'][:5]])
            
            # 执行多策略链接
            linking_results = self.enhanced_entity_linking.multi_strategy_linking(
                entities, context, umls_kg
            )
            
            return linking_results
            
        except Exception as e:
            logger.error(f"Error in enhanced entity linking: {e}")
            return {}
    
    def adaptive_knowledge_selection(self, question_types, entities):
        """
        自适应知识选择
        输入: question_types - 问题类型列表, entities - 实体列表
        返回: 选择的知识
        """
        # 检查是否启用自适应UMLS
        if not ABLATION_CONFIG['USE_ADAPTIVE_UMLS']:
            return []
            
        try:
            selected_knowledge = []
            
            # 为每种问题类型选择相关知识
            for question_type in question_types:
                knowledge = self.adaptive_umls_selector.select_relevant_umls_knowledge(
                    question_type, entities
                )
                selected_knowledge.extend(knowledge)
            
            return selected_knowledge
            
        except Exception as e:
            logger.error(f"Error in adaptive knowledge selection: {e}")
            return []

# ========================= 医学推理规则模块 =========================

class MedicalReasoningRules:
    """医学推理规则类 - 实现基于医学知识的推理规则"""
    def __init__(self, umls_normalizer=None):
        """
        初始化医学推理规则
        输入: umls_normalizer - UMLS标准化器
        """
        self.umls_normalizer = umls_normalizer  # UMLS标准化器
        self.kg_guided_reasoning = None         # KG引导推理器
        
        # 定义推理规则
        self.rules = {
            # 传递性规则
            'transitivity': {
                'causes': ['causes', 'leads_to', 'results_in', 'induces'],
                'treats': ['treats', 'alleviates', 'improves', 'cures'],
                'part_of': ['part_of', 'located_in', 'component_of'],
                'precedes': ['precedes', 'before', 'prior_to'],
                'prevents': ['prevents', 'reduces_risk_of', 'protects_against']
            },
            # 逆关系规则
            'inverse_relations': {
                'causes': 'caused_by',
                'treats': 'treated_by',
                'part_of': 'contains',
                'precedes': 'follows',
                'prevents': 'prevented_by'
            },
            # 语义蕴涵规则
            'semantic_implications': {
                'symptom_of': 'has_symptom',
                'risk_factor_for': 'has_risk_factor',
                'complication_of': 'has_complication'
            },
            # 医学层次规则
            'medical_hierarchies': {
                'disease_subtype': 'is_type_of',
                'anatomical_part': 'part_of_anatomy',
                'drug_class': 'belongs_to_class'
            }
        }
        
        # 置信度权重
        self.confidence_weights = {
            'direct': 1.0,           # 直接关系
            'transitive_1hop': 0.8,  # 一跳传递
            'transitive_2hop': 0.6,  # 两跳传递
            'inverse': 0.9,          # 逆关系
            'semantic': 0.7,         # 语义关系
            'hierarchical': 0.75     # 层次关系
        }
    
    def initialize_kg_guided_reasoning(self, kg, llm):
        """
        初始化知识图谱引导推理
        输入: kg - 知识图谱, llm - 语言模型
        """
        if ABLATION_CONFIG['USE_KG_GUIDED_REASONING']:
            self.kg_guided_reasoning = KGGuidedReasoningEngine(kg, llm)
        else:
            logger.info("🔬 KG-guided reasoning disabled in ablation study")
    
    def apply_reasoning_rules(self, knowledge_triples, max_hops=2):
        """
        应用医学推理规则扩展知识
        输入: knowledge_triples - 知识三元组列表, max_hops - 最大跳数
        返回: 扩展后的知识三元组列表
        """
        # 检查是否启用推理规则
        if not ABLATION_CONFIG['USE_REASONING_RULES']:
            logger.info("🔬 Medical reasoning rules disabled in ablation study")
            return knowledge_triples
            
        expanded_triples = knowledge_triples.copy()  # 复制原始三元组
        reasoning_log = []                           # 推理日志
        
        # 应用传递性规则
        transitive_triples = self._apply_transitivity(knowledge_triples, max_hops)
        expanded_triples.extend(transitive_triples)
        reasoning_log.extend([('transitivity', len(transitive_triples))])
        
        # 应用逆关系规则
        inverse_triples = self._apply_inverse_relations(knowledge_triples)
        expanded_triples.extend(inverse_triples)
        reasoning_log.extend([('inverse', len(inverse_triples))])
        
        # 应用语义蕴涵规则
        semantic_triples = self._apply_semantic_implications(knowledge_triples)
        expanded_triples.extend(semantic_triples)
        reasoning_log.extend([('semantic', len(semantic_triples))])
        
        # 应用层次推理规则
        hierarchical_triples = self._apply_hierarchical_reasoning(knowledge_triples)
        expanded_triples.extend(hierarchical_triples)
        reasoning_log.extend([('hierarchical', len(hierarchical_triples))])
        
        # 去重
        unique_triples = self._deduplicate_triples(expanded_triples)
        
        # 记录推理结果
        logger.info(f"推理扩展: {reasoning_log}")
        logger.info(f"原始三元组: {len(knowledge_triples)}, 扩展后: {len(unique_triples)}")
        
        return unique_triples
    
    def _apply_transitivity(self, triples, max_hops):
        """
        应用传递性推理
        输入: triples - 三元组列表, max_hops - 最大跳数
        返回: 传递性推理产生的新三元组
        """
        transitive_triples = []  # 存储传递性三元组
        
        # 构建关系图
        relation_graph = {}
        for triple in triples:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                if head not in relation_graph:
                    relation_graph[head] = []
                relation_graph[head].append((relation, tail))
        
        # 对每种传递性关系类型应用规则
        for rule_type, relation_variants in self.rules['transitivity'].items():
            transitive_triples.extend(
                self._find_transitive_paths(relation_graph, relation_variants, max_hops)
            )
        
        return transitive_triples
    
    def _find_transitive_paths(self, graph, relation_variants, max_hops):
        """
        查找传递性路径
        输入: graph - 关系图, relation_variants - 关系变体, max_hops - 最大跳数
        返回: 传递性路径列表
        """
        paths = []  # 存储路径
        
        # 遍历每个起始实体
        for start_entity in graph:
            # 对每个跳数进行搜索
            for hop in range(1, max_hops + 1):
                paths.extend(
                    self._dfs_transitive_search(graph, start_entity, relation_variants, hop, [])
                )
        
        return paths
    
    def _dfs_transitive_search(self, graph, current_entity, target_relations, remaining_hops, path):
        """
        深度优先搜索传递性路径
        输入: graph - 关系图, current_entity - 当前实体, target_relations - 目标关系,
             remaining_hops - 剩余跳数, path - 当前路径
        返回: 找到的路径列表
        """
        if remaining_hops == 0:
            return []
        
        results = []  # 存储结果
        
        if current_entity in graph:
            # 遍历当前实体的所有邻接实体
            for relation, next_entity in graph[current_entity]:
                # 检查关系是否匹配目标关系
                if any(target_rel in relation.lower() for target_rel in target_relations):
                    new_path = path + [(current_entity, relation, next_entity)]
                    
                    if remaining_hops == 1:
                        # 达到目标跳数，生成推理三元组
                        if len(new_path) >= 2:
                            start = new_path[0][0]
                            end = new_path[-1][2]
                            inferred_relation = f"transitively_{target_relations[0]}"
                            results.append([start, inferred_relation, end])
                    else:
                        # 继续搜索
                        results.extend(
                            self._dfs_transitive_search(
                                graph, next_entity, target_relations, 
                                remaining_hops - 1, new_path
                            )
                        )
        
        return results
    
    def _apply_inverse_relations(self, triples):
        """
        应用逆关系推理
        输入: triples - 三元组列表
        返回: 逆关系推理产生的新三元组
        """
        inverse_triples = []  # 存储逆关系三元组
        
        for triple in triples:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                # 查找匹配的逆关系
                for forward_rel, inverse_rel in self.rules['inverse_relations'].items():
                    if forward_rel in relation.lower():
                        # 生成逆关系三元组
                        inverse_triples.append([tail, inverse_rel, head])
        
        return inverse_triples
    
    def _apply_semantic_implications(self, triples):
        """
        应用语义蕴涵推理
        输入: triples - 三元组列表
        返回: 语义蕴涵推理产生的新三元组
        """
        semantic_triples = []  # 存储语义蕴涵三元组
        
        for triple in triples:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                # 查找匹配的语义蕴涵
                for source_rel, target_rel in self.rules['semantic_implications'].items():
                    if source_rel in relation.lower():
                        # 生成语义蕴涵三元组
                        semantic_triples.append([tail, target_rel, head])
        
        return semantic_triples
    
    def _apply_hierarchical_reasoning(self, triples):
        """
        应用层次推理
        输入: triples - 三元组列表
        返回: 层次推理产生的新三元组
        """
        hierarchical_triples = []  # 存储层次推理三元组
        
        if not self.umls_normalizer:
            return hierarchical_triples
        
        # 提取所有实体
        entities = set()
        for triple in triples:
            if len(triple) >= 3:
                entities.add(triple[0])
                entities.add(triple[2])
        
        # 为每个实体获取层次信息
        for entity in entities:
            try:
                hierarchy = self.umls_normalizer.get_concept_hierarchy(entity)
                
                # 生成上位概念关系
                for broader_concept in hierarchy.get('broader', []):
                    hierarchical_triples.append([entity, 'is_subtype_of', broader_concept])
                
                # 生成下位概念关系
                for narrower_concept in hierarchy.get('narrower', []):
                    hierarchical_triples.append([narrower_concept, 'is_subtype_of', entity])
                
            except Exception as e:
                logger.error(f"Error in hierarchical reasoning for {entity}: {e}")
        
        return hierarchical_triples
    
    def _deduplicate_triples(self, triples):
        """
        去重三元组
        输入: triples - 三元组列表
        返回: 去重后的三元组列表
        """
        seen = set()          # 已见过的三元组
        unique_triples = []   # 唯一三元组列表
        
        for triple in triples:
            if len(triple) >= 3:
                # 创建标准化的三元组键
                triple_key = (triple[0].lower(), triple[1].lower(), triple[2].lower())
                if triple_key not in seen:
                    seen.add(triple_key)
                    unique_triples.append(triple)
        
        return unique_triples

# ========================= 多跳推理模块 =========================

class MultiHopReasoning:
    """多跳推理类 - 实现复杂的多步骤推理"""
    def __init__(self, max_hops=3, umls_normalizer=None):
        """
        初始化多跳推理器
        输入: max_hops - 最大跳数, umls_normalizer - UMLS标准化器
        """
        self.max_hops = max_hops               # 最大跳数
        self.umls_normalizer = umls_normalizer # UMLS标准化器
        self.reasoning_chains = []             # 推理链
        
        # 证据权重
        self.evidence_weights = {
            'direct': 1.0,      # 直接证据
            'one_hop': 0.8,     # 一跳证据
            'two_hop': 0.6,     # 两跳证据
            'three_hop': 0.4    # 三跳证据
        }
        
        # 优化多跳推理器
        self.optimized_multi_hop = OptimizedMultiHopReasoning(kg=[], path_ranker=PathRanker())
    
    def perform_multi_hop_reasoning(self, question, kg_subgraph):
        """
        执行多跳推理
        输入: question - 问题, kg_subgraph - 知识图谱子图
        返回: 推理结果
        """
        # 检查是否启用优化多跳推理
        if not ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP']:
            return self._basic_multi_hop_reasoning(question, kg_subgraph)
            
        # 设置知识图谱
        self.optimized_multi_hop.kg = kg_subgraph
        
        # 提取问题中的实体
        question_entities = self._extract_question_entities(question)
        
        # UMLS标准化
        if self.umls_normalizer and ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            normalized_entities = self.umls_normalizer.normalize_medical_terms(question_entities)
            question_entities.extend(normalized_entities)
        
        # 构建推理链
        if len(question_entities) >= 2:
            # 多实体推理
            start_entities = question_entities[:1]
            target_entities = question_entities[1:]
            
            # 智能路径选择
            intelligent_paths = self.optimized_multi_hop.intelligent_path_selection(
                start_entities, target_entities, self.max_hops
            )
            
            # 构建推理链
            reasoning_chains = []
            for path in intelligent_paths[:5]:
                chain = self._build_reasoning_chain_from_path(path, kg_subgraph)
                if chain:
                    reasoning_chains.append(chain)
        else:
            # 单实体推理
            reasoning_chains = []
            for entity in question_entities:
                chain = self._build_reasoning_chain(entity, kg_subgraph, self.max_hops)
                if chain:
                    reasoning_chains.append(chain)
        
        # 融合推理链
        final_answer = self._fuse_reasoning_chains(reasoning_chains, question)
        return final_answer
    
    def _basic_multi_hop_reasoning(self, question, kg_subgraph):
        """基础版本的多跳推理（用于消融实验）"""
        logger.info("🔬 Using basic multi-hop reasoning (optimized version disabled)")
        
        # 提取问题实体
        question_entities = self._extract_question_entities(question)
        
        if len(question_entities) >= 2:
            reasoning_summary = f"Basic reasoning: Found entities {question_entities[:2]} in knowledge graph."
            
            # 查找直接连接
            direct_connections = []
            for triple in kg_subgraph:
                if len(triple) >= 3:
                    if (triple[0] in question_entities and triple[2] in question_entities) or \
                       (triple[2] in question_entities and triple[0] in question_entities):
                        direct_connections.append(f"{triple[0]} -> {triple[1]} -> {triple[2]}")
            
            if direct_connections:
                reasoning_summary += f" Found {len(direct_connections)} direct connections."
            
            return reasoning_summary
        else:
            return "Insufficient entities for multi-hop reasoning."
    
    def _build_reasoning_chain_from_path(self, path, kg_subgraph):
        """
        从路径构建推理链
        输入: path - 推理路径, kg_subgraph - 知识图谱子图
        返回: 推理链字典
        """
        chain = {
            'path': path,
            'confidence': self._calculate_path_confidence(path),
            'reasoning_steps': []
        }
        
        # 构建推理步骤
        for i, step in enumerate(path):
            if len(step) >= 3:
                reasoning_step = {
                    'step': i + 1,
                    'from': step[0],
                    'relation': step[1],
                    'to': step[2],
                    'explanation': f"Based on medical knowledge, {step[0]} {step[1]} {step[2]}"
                }
                chain['reasoning_steps'].append(reasoning_step)
        
        return chain
    
    def _calculate_path_confidence(self, path):
        """
        计算路径置信度
        输入: path - 路径
        返回: 置信度值
        """
        if not path:
            return 0.0
        
        total_confidence = 1.0
        # 计算每步的置信度
        for step in path:
            if len(step) >= 2:
                relation_weight = self._calculate_relation_weight(step[1])
                total_confidence *= relation_weight
        
        # 路径长度惩罚
        length_penalty = 0.9 ** len(path)
        return total_confidence * length_penalty
    
    def _extract_question_entities(self, question):
        """
        从问题中提取实体
        输入: question - 问题文本
        返回: 实体列表
        """
        entities = []  # 存储提取的实体
        
        # 医学术语列表
        medical_terms = [
            'alzheimer', 'dementia', 'brain', 'memory', 'cognitive',
            'treatment', 'medication', 'symptom', 'diagnosis', 'disease',
            'protein', 'amyloid', 'tau', 'hippocampus', 'cortex'
        ]
        
        question_lower = question.lower()
        # 查找医学术语
        for term in medical_terms:
            if term in question_lower:
                entities.append(term)
        
        # 提取大写单词（可能是专有名词）
        words = question.split()
        for word in words:
            if word[0].isupper() and len(word) > 3:
                entities.append(word)
        
        return list(set(entities))  # 去重返回
    
    def _build_reasoning_chain(self, start_entity, kg_subgraph, max_hops):
        """
        构建从起始实体开始的推理链
        输入: start_entity - 起始实体, kg_subgraph - 知识图谱子图, max_hops - 最大跳数
        返回: 推理链字典
        """
        chain = {
            'start_entity': start_entity,
            'paths': [],
            'confidence': 0.0
        }
        
        # 构建图结构
        graph = self._build_graph_from_subgraph(kg_subgraph)
        
        # 查找不同跳数的路径
        for hop in range(1, max_hops + 1):
            hop_paths = self._find_paths_at_hop(graph, start_entity, hop)
            chain['paths'].extend(hop_paths)
        
        # 计算推理链置信度
        chain['confidence'] = self._calculate_chain_confidence(chain['paths'])
        
        return chain
    
    def _build_graph_from_subgraph(self, kg_subgraph):
        """
        从子图构建图结构
        输入: kg_subgraph - 知识图谱子图
        返回: 图结构字典
        """
        graph = {}  # 存储图结构
        
        for triple in kg_subgraph:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                if head not in graph:
                    graph[head] = []
                
                # 添加边信息
                graph[head].append({
                    'relation': relation,
                    'target': tail,
                    'weight': self._calculate_relation_weight(relation)
                })
        
        return graph
    
    def _find_paths_at_hop(self, graph, start_entity, target_hop):
        """
        查找指定跳数的路径
        输入: graph - 图结构, start_entity - 起始实体, target_hop - 目标跳数
        返回: 路径列表
        """
        def dfs_path_search(current_entity, current_hop, path, visited):
            """深度优先搜索路径"""
            if current_hop == target_hop:
                return [path]
            
            if current_entity not in graph or current_entity in visited:
                return []
            
            visited.add(current_entity)
            paths = []
            
            # 遍历邻接节点
            for edge in graph[current_entity]:
                new_path = path + [(current_entity, edge['relation'], edge['target'])]
                paths.extend(
                    dfs_path_search(edge['target'], current_hop + 1, new_path, visited.copy())
                )
            
            return paths
        
        return dfs_path_search(start_entity, 0, [], set())
    
    def _calculate_relation_weight(self, relation):
        """
        计算关系权重
        输入: relation - 关系名称
        返回: 权重值
        """
        relation_lower = relation.lower().replace('_', ' ')
        
        # 关系权重表
        weights = {
            'causes': 3.0, 'treats': 2.8, 'prevents': 2.5,
            'associated_with': 2.2, 'diagnoses': 2.0,
            'symptom_of': 1.8, 'risk_factor': 1.6,
            'interacts_with': 1.4, 'located_in': 1.2,
            'part_of': 1.0, 'related_to': 0.8
        }
        
        # 查找匹配的权重
        for key, weight in weights.items():
            if key in relation_lower:
                return weight
        
        return 1.0  # 默认权重
    
    def _calculate_chain_confidence(self, paths):
        """
        计算推理链的置信度
        输入: paths - 路径列表
        返回: 置信度值
        """
        if not paths:
            return 0.0
        
        total_confidence = 0.0
        for path in paths:
            path_confidence = 1.0
            # 根据路径长度获取权重
            hop_weight = self.evidence_weights.get(f"{len(path)}_hop", 0.2)
            
            # 计算路径置信度
            for step in path:
                relation_weight = self._calculate_relation_weight(step[1])
                path_confidence *= relation_weight
            
            path_confidence *= hop_weight
            total_confidence += path_confidence
        
        return min(total_confidence / len(paths), 1.0)  # 归一化
    
    def _fuse_reasoning_chains(self, reasoning_chains, question):
        """
        融合推理结果
        输入: reasoning_chains - 推理链列表, question - 问题
        返回: 最终答案
        """
        if not reasoning_chains:
            return "Unable to find sufficient reasoning paths."
        
        # 按置信度排序
        reasoning_chains.sort(key=lambda x: x['confidence'], reverse=True)
        
        answer_components = []  # 答案组件
        total_confidence = 0.0  # 总置信度
        
        # 选择前3个高置信度的推理链
        for chain in reasoning_chains[:3]:
            if chain['confidence'] > 0.1:
                chain_summary = self._summarize_chain(chain)
                answer_components.append(chain_summary)
                total_confidence += chain['confidence']
        
        # 构建最终答案
        if answer_components:
            final_answer = f"Based on multi-hop reasoning (confidence: {total_confidence:.2f}):\n"
            final_answer += "\n".join(answer_components)
            return final_answer
        else:
            return "Insufficient evidence for multi-hop reasoning."
    
    def _summarize_chain(self, chain):
        """
        总结推理链
        输入: chain - 推理链
        返回: 推理链摘要
        """
        summary = f"From {chain['start_entity']}:"
        
        # 选择最佳路径
        best_paths = sorted(chain['paths'], 
                           key=lambda p: self._calculate_path_score(p), 
                           reverse=True)[:2]
        
        # 格式化路径
        for i, path in enumerate(best_paths):
            path_str = " -> ".join([f"{step[0]} ({step[1]}) {step[2]}" for step in path])
            summary += f"\nPath {i+1}: {path_str}"
        
        return summary
    
    def _calculate_path_score(self, path):
        """
        计算路径得分
        输入: path - 路径
        返回: 路径得分
        """
        score = 1.0
        for step in path:
            score *= self._calculate_relation_weight(step[1])
        return score / len(path)  # 平均得分

# ========================= 医学领域知识库 =========================

# 医学缩写词典
MEDICAL_ABBREVIATIONS = {
    'AD': 'Alzheimer Disease',
    'AIDS': 'Acquired Immunodeficiency Syndrome',
    'BP': 'Blood Pressure',
    'CAD': 'Coronary Artery Disease',
    'CHF': 'Congestive Heart Failure',
    'COPD': 'Chronic Obstructive Pulmonary Disease',
    'CVA': 'Cerebrovascular Accident',
    'DM': 'Diabetes Mellitus',
    'HTN': 'Hypertension',
    'MI': 'Myocardial Infarction',
    'MS': 'Multiple Sclerosis',
    'PD': "Parkinson's Disease",
    'PTSD': 'Post Traumatic Stress Disorder',
    'RA': 'Rheumatoid Arthritis',
    'TB': 'Tuberculosis',
    'UTI': 'Urinary Tract Infection',
    'HIV': 'Human Immunodeficiency Virus',
    'CT': 'Computed Tomography',
    'MRI': 'Magnetic Resonance Imaging',
    'EEG': 'Electroencephalogram',
    'ECG': 'Electrocardiogram',
    'CSF': 'Cerebrospinal Fluid',
    'MCI': 'Mild Cognitive Impairment',
    'NFT': 'Neurofibrillary Tangles',
    'APP': 'Amyloid Precursor Protein',
    'APOE': 'Apolipoprotein E',
    'AChE': 'Acetylcholinesterase',
    'MMSE': 'Mini Mental State Examination'
}

# 医学同义词典
MEDICAL_SYNONYMS = {
    'alzheimer': ['alzheimer disease', 'dementia', 'alzheimers', 'ad', 'alzheimer\'s disease'],
    'heart attack': ['myocardial infarction', 'mi', 'cardiac arrest'],
    'stroke': ['cerebrovascular accident', 'cva', 'brain attack'],
    'high blood pressure': ['hypertension', 'htn', 'elevated bp'],
    'diabetes': ['diabetes mellitus', 'dm', 'diabetic'],
    'cancer': ['carcinoma', 'tumor', 'malignancy', 'neoplasm'],
    'infection': ['infectious disease', 'sepsis', 'inflammation'],
    'treatment': ['therapy', 'medication', 'drug', 'medicine'],
    'symptom': ['sign', 'manifestation', 'presentation'],
    'diagnosis': ['diagnostic', 'identification', 'detection'],
    'cognitive decline': ['cognitive impairment', 'dementia', 'memory loss'],
    'memory problems': ['memory deficit', 'memory loss', 'amnesia'],
    'confusion': ['disorientation', 'bewilderment', 'perplexity'],
    'brain imaging': ['neuroimaging', 'brain scan', 'ct scan', 'mri scan']
}

# 关系重要性权重
RELATION_IMPORTANCE_WEIGHTS = {
    'cause': 3.0, 'causes': 3.0, 'caused_by': 3.0,
    'treat': 2.8, 'treats': 2.8, 'treatment': 2.8,
    'prevent': 2.5, 'prevents': 2.5, 'prevention': 2.5,
    'associate': 2.2, 'associates': 2.2, 'associated_with': 2.2,
    'diagnose': 2.0, 'diagnosis': 2.0, 'diagnostic': 2.0,
    'symptom': 1.8, 'symptoms': 1.8, 'has_symptom': 1.8,
    'risk_factor': 1.6, 'risk': 1.6,
    'interact': 1.4, 'interacts': 1.4, 'interaction': 1.4,
    'located_in': 1.2, 'location': 1.2,
    'part_of': 1.0, 'is_a': 1.0,
    'related': 0.8, 'similar': 0.6
}

# 问题类型关键词
QUESTION_TYPE_KEYWORDS = {
    'definition': ['what is', 'define', 'definition', 'meaning'],
    'causation': ['cause', 'causes', 'reason', 'why', 'due to', 'because'],
    'treatment': ['treat', 'treatment', 'therapy', 'cure', 'medication', 'drug'],
    'symptom': ['symptom', 'sign', 'present', 'manifestation'],
    'diagnosis': ['diagnose', 'diagnosis', 'test', 'examination'],
    'prevention': ['prevent', 'prevention', 'avoid', 'reduce risk'],
    'exception': ['except', 'not', 'exclude', 'excluding', 'other than']
}

# 否定词
NEGATION_WORDS = ['not', 'except', 'excluding', 'other than', 'rather than', 'instead of', 'exclude']

# 数据集处理器映射
dataset2processor = {
    'medmcqa': medmcqaZeroshotsProcessor,
    'medqa':medqaZeroshotsProcessor,
    'mmlu': mmluZeroshotsProcessor,
    'qa4mre':qa4mreZeroshotsProcessor
}
datasets = ['medqa', 'medmcqa', 'mmlu', 'qa4mre']

# ========================= 初始化增强模块 =========================
# UMLS API密钥
umls_api_key = "7cce913d-29bf-459f-aa9a-2ba57d6efccf"
# 创建UMLS标准化器
umls_normalizer = UMLSNormalizer(umls_api_key)
# 创建医学推理规则实例
medical_reasoning_rules = MedicalReasoningRules(umls_normalizer)
# 创建多跳推理器
multi_hop_reasoner = MultiHopReasoning(max_hops=3, umls_normalizer=umls_normalizer)
# 创建层次化知识图谱框架
hierarchical_kg_framework = HierarchicalKGFramework()

# ========================= 性能优化函数 =========================

def cleanup_resources(sample_count):
    """
    性能优化：定期清理系统资源
    输入: sample_count - 当前处理的样本数量
    """
    try:
        # 执行垃圾回收
        collected = gc.collect()
        
        # 清理UMLS API缓存
        if hasattr(umls_normalizer, 'umls_api') and hasattr(umls_normalizer.umls_api, 'cache'):
            cache_size_before = len(umls_normalizer.umls_api.cache)
            if cache_size_before > MAX_CACHE_SIZE:
                # 保留最近的缓存项
                cache_items = list(umls_normalizer.umls_api.cache.items())
                umls_normalizer.umls_api.cache = dict(cache_items[-KEEP_CACHE_SIZE:])
                logger.info(f"🧹 Cleaned UMLS cache: {cache_size_before} → {len(umls_normalizer.umls_api.cache)}")
        
        # 清理本地缓存
        if hasattr(umls_normalizer, 'local_cache'):
            local_cache_size_before = len(umls_normalizer.local_cache)
            if local_cache_size_before > MAX_CACHE_SIZE:
                cache_items = list(umls_normalizer.local_cache.items())
                umls_normalizer.local_cache = dict(cache_items[-KEEP_CACHE_SIZE:])
                logger.info(f"🧹 Cleaned local cache: {local_cache_size_before} → {len(umls_normalizer.local_cache)}")
        
        # 清理失败CUI缓存
        if hasattr(umls_normalizer, 'umls_api') and hasattr(umls_normalizer.umls_api, 'failed_cuis'):
            failed_cuis_size_before = len(umls_normalizer.umls_api.failed_cuis)
            if failed_cuis_size_before > MAX_FAILED_CUIS:
                umls_normalizer.umls_api.failed_cuis.clear()
                logger.info(f"🧹 Cleaned failed CUI cache: {failed_cuis_size_before} → 0")
        
        # 清理推理缓存
        if hasattr(multi_hop_reasoner, 'optimized_multi_hop') and hasattr(multi_hop_reasoner.optimized_multi_hop, 'reasoning_cache'):
            reasoning_cache_size_before = len(multi_hop_reasoner.optimized_multi_hop.reasoning_cache)
            if reasoning_cache_size_before > 500:
                multi_hop_reasoner.optimized_multi_hop.reasoning_cache.clear()
                logger.info(f"🧹 Cleaned reasoning cache: {reasoning_cache_size_before} → 0")
        
        logger.info(f"✅ Resource cleanup completed at sample {sample_count} (collected {collected} objects)")
        
    except Exception as e:
        logger.error(f"❌ Error during resource cleanup: {e}")

# ========================= 核心功能函数 =========================

def expand_medical_abbreviations(text):
    """
    扩展医学缩写词
    输入: text - 包含缩写的文本
    返回: 扩展缩写后的文本
    """
    expanded_text = text
    # 遍历所有缩写词进行替换
    for abbr, full_form in MEDICAL_ABBREVIATIONS.items():
        # 使用正则表达式进行精确匹配（单词边界）
        pattern = r'\b' + re.escape(abbr) + r'\b'
        expanded_text = re.sub(pattern, full_form, expanded_text, flags=re.IGNORECASE)
    return expanded_text

def get_medical_synonyms(entity):
    """
    获取医学术语的同义词
    输入: entity - 医学实体
    返回: 同义词列表
    """
    entity_lower = entity.lower()
    synonyms = [entity]  # 包含原始实体
    
    # 如果启用UMLS标准化，获取UMLS变体
    if ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
        try:
            umls_variants = umls_normalizer.get_semantic_variants(entity)
            synonyms.extend(umls_variants)
            logger.debug(f"UMLS variants for '{entity}': {umls_variants}")
        except Exception as e:
            logger.error(f"Error getting UMLS variants for '{entity}': {e}")
    
    # 从医学同义词词典中查找
    for key, synonym_list in MEDICAL_SYNONYMS.items():
        if key in entity_lower or entity_lower in synonym_list:
            synonyms.extend(synonym_list)
    
    # 对同义词进行UMLS标准化
    if ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
        try:
            normalized_synonyms = umls_normalizer.normalize_medical_terms(synonyms)
            synonyms.extend(normalized_synonyms)
        except Exception as e:
            logger.error(f"Error normalizing synonyms for '{entity}': {e}")
    
    return list(set(synonyms))  # 去重返回

def identify_question_type(question):
    """
    识别问题类型
    输入: question - 问题文本
    返回: 问题类型列表
    """
    question_lower = question.lower()
    question_types = []
    
    # 遍历问题类型关键词进行匹配
    for q_type, keywords in QUESTION_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in question_lower:
                question_types.append(q_type)
                break  # 找到一个匹配后跳出内层循环
    
    return question_types if question_types else ['general']  # 默认为通用类型

def has_negation(question):
    """
    检查问题是否包含否定词
    输入: question - 问题文本
    返回: 是否包含否定词
    """
    question_lower = question.lower()
    return any(neg_word in question_lower for neg_word in NEGATION_WORDS)

def calculate_relation_weight(relation_type):
    """
    计算关系重要性权重
    输入: relation_type - 关系类型
    返回: 权重值
    """
    relation_lower = relation_type.lower().replace('_', ' ')
    
    # 直接查找权重
    if relation_lower in RELATION_IMPORTANCE_WEIGHTS:
        return RELATION_IMPORTANCE_WEIGHTS[relation_lower]
    
    # 模糊匹配
    for key, weight in RELATION_IMPORTANCE_WEIGHTS.items():
        if key in relation_lower or relation_lower in key:
            return weight
    
    return 1.0  # 默认权重

def calculate_knowledge_quality_score(knowledge_items):
    """
    计算知识质量分数
    输入: knowledge_items - 知识项列表
    返回: 质量分数
    """
    if not knowledge_items:
        return 0.0
    
    quality_scores = []
    
    for item in knowledge_items:
        score = 1.0  # 基础分数
        
        if isinstance(item, list) and len(item) >= 3:
            entity, relation, objects = item[0], item[1], item[2]
            
            # 实体长度加分
            if len(entity) > 3:
                score += 0.5
            
            # 关系权重加分
            relation_weight = calculate_relation_weight(relation)
            score += relation_weight * 0.3
            
            # 对象数量加分
            object_count = len(objects.split(',')) if ',' in objects else 1
            score += min(object_count * 0.1, 1.0)
        
        quality_scores.append(score)
    
    return np.mean(quality_scores)  # 返回平均质量分数

def convert_numpy_types(obj):
    """
    递归转换NumPy类型为Python原生类型
    输入: obj - 待转换对象
    返回: 转换后的对象
    """
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def retry_on_failure(max_retries=MAX_RETRIES, wait_time=RETRY_WAIT_TIME):
    """
    添加重试机制的装饰器
    输入: max_retries - 最大重试次数, wait_time - 等待时间
    返回: 装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 尝试执行函数
            for retry in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    # 检查结果是否有效
                    if result is not None and result != "":
                        return result
                    else:
                        logger.warning(f"Function {func.__name__} returned empty result, retrying {retry + 1}/{max_retries}...")
                        sleep(wait_time)
                except Exception as e:
                    logger.error(f"Function {func.__name__} failed attempt {retry + 1}/{max_retries}: {e}")
                    sleep(wait_time)
            
            logger.error(f"Function {func.__name__} exhausted all retries, returning empty string")
            return ""
        return wrapper
    return decorator

def chat_35(prompt):
    """
    GPT-3.5 Turbo聊天接口函数
    输入: prompt - 提示文本
    返回: GPT-3.5的响应内容
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # 使用GPT-3.5 Turbo模型
        messages=[
            {"role": "user", "content": prompt}  # 用户角色消息
        ])
    return completion.choices[0].message.content  # 返回第一个选择的消息内容

def chat_4(prompt):
    """
    GPT-4聊天接口函数
    输入: prompt - 提示文本
    返回: GPT-4的响应内容
    """
    completion = openai.ChatCompletion.create(
        model="gpt-4",  # 使用GPT-4模型
        messages=[
            {"role": "user", "content": prompt}  # 用户角色消息
        ])
    return completion.choices[0].message.content  # 返回第一个选择的消息内容

def validate_knowledge_triple(head, relation, tail):
    """
    验证知识三元组的质量
    检查三元组是否有效、完整且不含噪声
    输入: head - 头实体, relation - 关系, tail - 尾实体
    返回: 是否为有效三元组
    """
    # 检查是否存在空值
    if pd.isna(head) or pd.isna(relation) or pd.isna(tail):
        return False
    
    # 转换为字符串并去除空白
    head = str(head).strip() if head is not None else ""
    relation = str(relation).strip() if relation is not None else ""
    tail = str(tail).strip() if tail is not None else ""
    
    # 检查是否为空字符串
    if not head or not relation or not tail:
        return False
    
    # 检查最小长度要求
    if len(head) < 2 or len(tail) < 2:
        return False
    
    # 检查噪声模式
    noise_patterns = ['http', 'www', '@', '#', '___', '...', 'nan', 'none']
    for pattern in noise_patterns:
        if pattern in head.lower() or pattern in tail.lower():
            return False
    
    return True  # 通过所有检查

def basic_entity_matching(question_kg, entity_embeddings, keyword_embeddings, question_text=""):
    """
    基础版本的实体匹配，用于消融实验
    当增强功能被禁用时使用的简单匹配策略
    输入: question_kg - 问题中的实体, entity_embeddings - 实体嵌入, 
         keyword_embeddings - 关键词嵌入, question_text - 问题文本
    返回: 匹配的实体列表和置信度分数
    """
    match_kg = []                    # 匹配的实体列表
    entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])  # 转为DataFrame
    entity_confidence_scores = []    # 置信度分数列表
    
    # 遍历问题中的每个实体
    for kg_entity in question_kg:
        try:
            # 在关键词嵌入中查找实体
            if kg_entity in keyword_embeddings["keywords"]:
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
            else:
                # 模糊匹配
                best_match_idx = None
                best_similarity = 0
                for idx, keyword in enumerate(keyword_embeddings["keywords"]):
                    if kg_entity.lower() in keyword.lower():
                        similarity = 0.8  # 简单的相似度分数
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_idx = idx
                
                if best_match_idx is None:
                    continue  # 没有找到匹配，跳过
                keyword_index = best_match_idx
            
            # 获取实体嵌入向量
            kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

            # 归一化嵌入向量
            kg_entity_emb_norm = kg_entity_emb / np.linalg.norm(kg_entity_emb)
            entity_embeddings_norm = entity_embeddings_emb.values / np.linalg.norm(entity_embeddings_emb.values, axis=1, keepdims=True)
            
            # 计算余弦相似度
            cos_similarities = np.dot(entity_embeddings_norm, kg_entity_emb_norm)
            
            # 找到最佳匹配
            best_idx = np.argmax(cos_similarities)
            similarity_score = cos_similarities[best_idx]
            
            # 检查相似度阈值
            if similarity_score >= 0.6:
                candidate_entity = entity_embeddings["entities"][best_idx]
                if candidate_entity not in match_kg:  # 避免重复
                    match_kg.append(candidate_entity)
                    entity_confidence_scores.append(float(similarity_score))
                    logger.debug(f"Basic matched: {kg_entity} -> {candidate_entity} (score: {similarity_score:.3f})")
                
        except Exception as e:
            logger.error(f"Error in basic entity matching for {kg_entity}: {e}")
            continue
    
    return match_kg, entity_confidence_scores

def enhanced_entity_matching(question_kg, entity_embeddings, keyword_embeddings, question_text=""):
    """
    增强的实体匹配，集成真实UMLS API和新优化
    结合多种策略进行更准确的实体匹配
    输入: question_kg - 问题实体, entity_embeddings - 实体嵌入, 
         keyword_embeddings - 关键词嵌入, question_text - 问题文本
    返回: 匹配的实体列表和置信度分数
    """
    
    # 检查是否有增强功能被启用
    if not any([
        ABLATION_CONFIG['USE_HIERARCHICAL_KG'],
        ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING'],
        ABLATION_CONFIG['USE_ADAPTIVE_UMLS'],
        ABLATION_CONFIG['USE_UMLS_NORMALIZATION'],
        ABLATION_CONFIG['USE_REASONING_RULES']
    ]):
        logger.info("🔬 Using basic entity matching (all enhancements disabled)")
        return basic_entity_matching(question_kg, entity_embeddings, keyword_embeddings, question_text)
    
    match_kg = []                    # 匹配的实体列表
    entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
    entity_confidence_scores = []    # 置信度分数
    
    # 识别问题类型
    question_types = identify_question_type(question_text)
    
    # 扩展实体列表
    expanded_entities = []
    for kg_entity in question_kg:
        # 扩展医学缩写
        expanded_entity = expand_medical_abbreviations(kg_entity)
        expanded_entities.append(expanded_entity)
        
        # 如果启用UMLS标准化，获取同义词
        if ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            synonyms = get_medical_synonyms(kg_entity)
            expanded_entities.extend(synonyms)
    
    # 多策略实体链接
    if ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
        try:
            enhanced_links = umls_normalizer.enhanced_entity_linking_method(
                expanded_entities, question_text, question_types
            )
            
            # 添加高分链接的实体
            for entity, link_info in enhanced_links.items():
                if link_info.get('final_score', 0) > 0.6:
                    expanded_entities.append(entity)
                    
        except Exception as e:
            logger.error(f"Error in enhanced entity linking: {e}")
    
    # 自适应UMLS知识选择
    if ABLATION_CONFIG['USE_ADAPTIVE_UMLS']:
        try:
            adaptive_knowledge = umls_normalizer.adaptive_knowledge_selection(
                question_types, expanded_entities
            )
            
            # 从自适应知识中提取实体
            for knowledge_item in adaptive_knowledge:
                if isinstance(knowledge_item, dict):
                    related_name = knowledge_item.get('related_name', '')
                    if related_name:
                        expanded_entities.append(related_name)
                        
        except Exception as e:
            logger.error(f"Error in adaptive knowledge selection: {e}")
    
    # 推理规则扩展
    if ABLATION_CONFIG['USE_REASONING_RULES']:
        try:
            # 创建临时三元组用于推理
            temp_triples = [[entity, 'mentions', 'question'] for entity in expanded_entities]
            reasoned_triples = medical_reasoning_rules.apply_reasoning_rules(temp_triples)
            
            # 从推理结果中提取实体
            for triple in reasoned_triples:
                if len(triple) >= 3:
                    expanded_entities.extend([triple[0], triple[2]])
        except Exception as e:
            logger.error(f"Error in reasoning-based entity expansion: {e}")
    
    # 去重
    seen = set()
    unique_entities = []
    for entity in expanded_entities:
        if entity.lower() not in seen:
            seen.add(entity.lower())
            unique_entities.append(entity)
    
    logger.info(f"Original entities: {question_kg}")
    logger.info(f"Expanded entities (with optimizations): {unique_entities[:10]}...")
    
    # 根据问题类型调整相似度阈值
    is_negation = has_negation(question_text)
    if 'exception' in question_types or is_negation:
        similarity_threshold = MIN_SIMILARITY_THRESHOLD * 0.8  # 降低阈值
    else:
        similarity_threshold = MIN_SIMILARITY_THRESHOLD
    
    # 对每个扩展后的实体进行匹配
    for kg_entity in unique_entities:
        try:
            # 在关键词嵌入中查找
            if kg_entity in keyword_embeddings["keywords"]:
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
            else:
                # 改进的模糊匹配
                best_match_idx = None
                best_similarity = 0
                for idx, keyword in enumerate(keyword_embeddings["keywords"]):
                    if kg_entity.lower() in keyword.lower() or keyword.lower() in kg_entity.lower():
                        # 计算Jaccard相似度
                        similarity = len(set(kg_entity.lower().split()) & set(keyword.lower().split())) / len(set(kg_entity.lower().split()) | set(keyword.lower().split()))
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_idx = idx
                
                if best_match_idx is None or best_similarity < 0.3:
                    continue  # 没有足够好的匹配
                keyword_index = best_match_idx
            
            # 获取实体嵌入
            kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

            # 归一化
            kg_entity_emb_norm = kg_entity_emb / np.linalg.norm(kg_entity_emb)
            entity_embeddings_norm = entity_embeddings_emb.values / np.linalg.norm(entity_embeddings_emb.values, axis=1, keepdims=True)
            
            # 计算余弦相似度
            cos_similarities = np.dot(entity_embeddings_norm, kg_entity_emb_norm)
            
            # 获取前5个候选
            top_indices = np.argsort(cos_similarities)[::-1]
            
            # 寻找最佳匹配
            best_match_found = False
            for idx in top_indices[:5]:
                similarity_score = cos_similarities[idx]
                candidate_entity = entity_embeddings["entities"][idx]
                
                if (similarity_score >= similarity_threshold and 
                    candidate_entity not in match_kg):
                    match_kg.append(candidate_entity)
                    entity_confidence_scores.append(float(similarity_score))
                    best_match_found = True
                    logger.debug(f"Matched: {kg_entity} -> {candidate_entity} (score: {similarity_score:.3f})")
                    break
            
            if not best_match_found:
                logger.warning(f"No high-confidence match found for entity: {kg_entity}")
                
        except (ValueError, IndexError):
            logger.error(f"Entity {kg_entity} not found in keyword embeddings")
            continue
        except Exception as e:
            logger.error(f"Error processing entity {kg_entity}: {e}")
            continue
    
    # 记录平均置信度
    if entity_confidence_scores:
        avg_confidence = np.mean(entity_confidence_scores)
        logger.info(f"Entity matching average confidence: {avg_confidence:.3f}")
    
    return match_kg, entity_confidence_scores

def enhanced_find_shortest_path(start_entity_name, end_entity_name, candidate_list, question_types=[]):
    """
    增强的路径查找，带有医学知识权重
    在Neo4j知识图谱中查找两个实体之间的最短路径
    输入: start_entity_name - 起始实体名, end_entity_name - 结束实体名, 
         candidate_list - 候选实体列表, question_types - 问题类型
    返回: 路径列表和存在的实体
    """
    global exist_entity  # 全局变量，存储存在的实体
    paths_with_scores = []  # 存储带分数的路径
    
    # 使用Neo4j会话
    with driver.session() as session:
        try:
            # 查找所有最短路径的Cypher查询
            result = session.run(
                "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
                "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "  # 最多5跳
                "RETURN p LIMIT 15",  # 限制返回15条路径
                start_entity_name=start_entity_name,
                end_entity_name=end_entity_name
            )
            
            paths = []       # 存储路径
            short_path = 0   # 短路径标志
            
            # 处理查询结果
            for record in result:
                path = record["p"]  # 获取路径
                entities = []       # 路径中的实体
                relations = []      # 路径中的关系
                path_quality_score = 0  # 路径质量分数
                
                # 提取路径中的节点和关系
                for i in range(len(path.nodes)):
                    node = path.nodes[i]
                    entity_name = node["name"]
                    entities.append(entity_name)
                    
                    # 处理关系
                    if i < len(path.relationships):
                        relationship = path.relationships[i]
                        relation_type = relationship.type
                        relations.append(relation_type)
                        
                        # 计算关系权重
                        if any([ABLATION_CONFIG['USE_HIERARCHICAL_KG'], 
                               ABLATION_CONFIG['USE_REASONING_RULES'],
                               ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP']]):
                            relation_weight = calculate_relation_weight(relation_type)
                            path_quality_score += relation_weight
                            
                            # 根据问题类型调整分数
                            if question_types:
                                if 'treatment' in question_types and 'treat' in relation_type.lower():
                                    path_quality_score += 1.0
                                elif 'causation' in question_types and 'cause' in relation_type.lower():
                                    path_quality_score += 1.0
                                elif 'symptom' in question_types and 'symptom' in relation_type.lower():
                                    path_quality_score += 1.0
                        else:
                            path_quality_score += 1.0  # 基础分数
               
                # 构建路径字符串
                path_str = ""
                for i in range(len(entities)):
                    entities[i] = entities[i].replace("_"," ")  # 替换下划线
                    
                    # 检查实体是否在候选列表中
                    if entities[i] in candidate_list:
                        short_path = 1  # 找到短路径
                        exist_entity = entities[i]
                        path_quality_score += 3  # 候选实体加分
                        
                    path_str += entities[i]
                    if i < len(relations):
                        relations[i] = relations[i].replace("_"," ")
                        path_str += "->" + relations[i] + "->"
                
                # 路径长度惩罚
                path_length = len(relations)
                length_penalty = path_length * 0.1 if ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP'] else 0
                final_score = path_quality_score - length_penalty
                
                paths_with_scores.append((path_str, final_score))
                
                # 如果找到短路径，优先返回
                if short_path == 1:
                    if ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP']:
                        paths_with_scores.sort(key=lambda x: x[1], reverse=True)
                    paths = [path[0] for path in paths_with_scores[:5]]
                    break
            
            # 如果没有短路径，返回最佳路径
            if not paths and paths_with_scores:
                if ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP']:
                    paths_with_scores.sort(key=lambda x: x[1], reverse=True)
                paths = [path[0] for path in paths_with_scores[:5]]
                exist_entity = {}
            
        except Exception as e:
            logger.error(f"Error in path finding: {e}")
            return [], {}
            
        try:
            return paths, exist_entity
        except:
            return paths, {}

def find_shortest_path(start_entity_name, end_entity_name, candidate_list, question_types=[]):
    """
    原始函数，使用增强实现
    为了保持向后兼容性而保留的函数签名
    """
    return enhanced_find_shortest_path(start_entity_name, end_entity_name, candidate_list, question_types)

def combine_lists(*lists):
    """
    组合多个列表的所有可能组合
    使用笛卡尔积生成组合
    输入: *lists - 可变数量的列表
    返回: 所有可能的组合列表
    """
    combinations = list(itertools.product(*lists))  # 计算笛卡尔积
    results = []
    
    # 处理每个组合
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist  # 展平列表
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    
    return results

def enhanced_get_entity_neighbors(entity_name: str, disease_flag, question_types=[]) -> Tuple[List[List[str]], List[str]]:
    """
    增强的邻居提取，带有问题类型感知过滤
    从Neo4j图数据库中获取实体的邻居节点
    输入: entity_name - 实体名称, disease_flag - 疾病标志, question_types - 问题类型列表
    返回: 邻居列表和疾病列表的元组
    """
    disease = []        # 疾病列表
    neighbor_list = []  # 邻居列表
    
    # 根据问题类型和启用的功能调整限制
    if any([ABLATION_CONFIG['USE_ADAPTIVE_UMLS'], ABLATION_CONFIG['USE_REASONING_RULES']]):
        limit = 25 if any(q_type in ['treatment', 'causation'] for q_type in question_types) else 20
    else:
        limit = 10  # 基础限制
    
    # 构建Cypher查询
    query = f"""
    MATCH (e:Entity)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type,
           collect(n.name) AS neighbor_entities
    ORDER BY size(collect(n.name)) DESC
    LIMIT {limit}
    """
    
    try:
        # 执行查询
        result = session.run(query, entity_name=entity_name)
        relation_quality_scores = {}  # 关系质量分数
        
        # 处理查询结果
        for record in result:
            rel_type = record["relationship_type"]
            
            # 疾病标志过滤
            if disease_flag == 1 and rel_type == 'has_symptom':
                continue  # 跳过症状关系

            neighbors = record["neighbor_entities"]
            
            # 计算关系质量分数
            if ABLATION_CONFIG['USE_REASONING_RULES']:
                quality_score = calculate_relation_weight(rel_type)
                
                # 根据问题类型调整分数
                if question_types:
                    if 'treatment' in question_types and 'treat' in rel_type.lower():
                        quality_score += 1.0
                    elif 'causation' in question_types and 'cause' in rel_type.lower():
                        quality_score += 1.0
                    elif 'symptom' in question_types and 'symptom' in rel_type.lower():
                        quality_score += 1.0
            else:
                quality_score = 1.0  # 默认分数
            
            # 疾病关系特殊处理
            if "disease" in rel_type.replace("_"," ").lower():
                disease.extend(neighbors)
                quality_score += 1.0  # 疾病关系加分
                
            # 过滤有效邻居
            filtered_neighbors = []
            for neighbor in neighbors:
                if validate_knowledge_triple(entity_name, rel_type, neighbor):
                    filtered_neighbors.append(neighbor)
            
            # 添加到邻居列表
            if filtered_neighbors:
                neighbor_entry = [entity_name.replace("_"," "), rel_type.replace("_"," "), 
                                ','.join([x.replace("_"," ") for x in filtered_neighbors])]
                neighbor_list.append(neighbor_entry)
                relation_quality_scores[len(neighbor_list)-1] = quality_score
        
        # 根据质量分数排序
        if relation_quality_scores and ABLATION_CONFIG['USE_REASONING_RULES']:
            sorted_indices = sorted(relation_quality_scores.keys(), 
                                  key=lambda k: relation_quality_scores[k], reverse=True)
            neighbor_list = [neighbor_list[i] for i in sorted_indices]
    
    except Exception as e:
        logger.error(f"Error getting entity neighbors: {e}")
    
    return neighbor_list, disease

def get_entity_neighbors(entity_name: str, disease_flag, question_types=[]) -> List[List[str]]:
    """
    原始函数签名保持不变
    为了保持向后兼容性而保留
    """
    neighbor_list, disease = enhanced_get_entity_neighbors(entity_name, disease_flag, question_types)
    return neighbor_list, disease

@retry_on_failure()
def prompt_path_finding(path_input):
    """
    原始路径查找提示模板
    将知识图谱路径转换为自然语言描述
    输入: path_input - 路径输入字符串
    返回: 自然语言描述的路径证据
    """
    # 路径转换为自然语言的提示模板
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

    # 调用聊天模型
    response = chat(chat_prompt_with_values.to_messages())
    if response.content is not None:
        return response.content
    else:
        return ""

@retry_on_failure()
def prompt_neighbor(neighbor):
    """
    原始邻居提示模板
    将知识图谱邻居信息转换为自然语言描述
    输入: neighbor - 邻居信息字符串
    返回: 自然语言描述的邻居证据
    """
    # 邻居转换为自然语言的提示模板
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

    # 调用聊天模型
    response = chat(chat_prompt_with_values.to_messages())
    if response.content is not None:
        return response.content
    else:
        return ""

@retry_on_failure()
def self_knowledge_retrieval(graph, question):
    """
    原始知识检索提示模板
    从知识图谱中过滤与问题相关的知识
    输入: graph - 知识图谱字符串, question - 问题文本
    返回: 过滤后的知识
    """
    # 知识过滤的提示模板
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

    # 调用聊天模型
    response = chat(chat_prompt_with_values.to_messages())
    if response.content is not None:
        return response.content
    else:
        return ""

def enhanced_self_knowledge_retrieval_reranking(graph, question):
    """
    增强的知识重排序，带有医学知识感知和多跳推理
    根据问题类型和医学知识对知识图谱进行智能重排序
    输入: graph - 知识图谱字符串, question - 问题文本
    返回: 重排序后的知识
    """
    
    # 检查是否启用增强功能
    if not any([ABLATION_CONFIG['USE_REASONING_RULES'], 
               ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP'],
               ABLATION_CONFIG['USE_KG_GUIDED_REASONING']]):
        logger.info("🔬 Using basic knowledge retrieval reranking")
        return self_knowledge_retrieval(graph, question)
    
    # 识别问题特征
    question_types = identify_question_type(question)
    has_neg = has_negation(question)
    
    # 尝试多跳推理
    if ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP']:
        try:
            # 解析知识图谱为三元组
            graph_triples = []
            for line in graph.split('\n'):
                if '->' in line:
                    parts = line.split('->')
                    if len(parts) >= 3:
                        head = parts[0].strip()
                        relation = parts[1].strip()
                        tail = '->'.join(parts[2:]).strip()
                        graph_triples.append([head, relation, tail])
            
            # 执行多跳推理
            if graph_triples:
                reasoned_result = multi_hop_reasoner.perform_multi_hop_reasoning(question, graph_triples)
                logger.debug(f"Multi-hop reasoning result: {reasoned_result[:200]}...")
        
        except Exception as e:
            logger.error(f"Error in multi-hop reasoning during reranking: {e}")
    
    # 根据问题类型生成专门的指令
    if 'exception' in question_types or has_neg:
        focus_instruction = "Focus on triples that help identify what is NOT associated with the question topic or what should be excluded."
    elif 'treatment' in question_types:
        focus_instruction = "Focus on triples related to treatments, therapies, medications, and therapeutic interventions."
    elif 'causation' in question_types:
        focus_instruction = "Focus on triples showing causal relationships, risk factors, and etiological connections."
    elif 'symptom' in question_types:
        focus_instruction = "Focus on triples describing symptoms, signs, manifestations, and clinical presentations."
    else:
        focus_instruction = "Focus on triples that directly relate to the question entities and provide meaningful medical relationships."
    
    # 构建增强的提示模板
    template = f"""
    There is a question and some knowledge graph. The knowledge graphs follow entity->relationship->entity list format.
    \n\n
    ##Graph: {{graph}}
    \n\n
    ##Question: {{question}}
    \n\n
    Please rerank the knowledge graph and output at most 5 important and relevant triples for solving the given question. {focus_instruction} Output the reranked knowledge in the following format:
    Reranked Triple1: xxx ——> xxx
    Reranked Triple2: xxx ——> xxx
    Reranked Triple3: xxx ——> xxx
    Reranked Triple4: xxx ——> xxx
    Reranked Triple5: xxx ——> xxx

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

    # 多次尝试调用
    for attempt in range(3):
        try:
            response = chat(chat_prompt_with_values.to_messages())
            if response.content is not None and len(response.content.strip()) > 10:
                return response.content
            else:
                logger.warning(f"Reranking attempt {attempt + 1} returned insufficient content")
                sleep(5)
        except Exception as e:
            logger.error(f"Reranking attempt {attempt + 1} failed: {e}")
            sleep(5)
    
    logger.error("All reranking attempts failed, returning original graph")
    return graph

@retry_on_failure()
def self_knowledge_retrieval_reranking(graph, question):
    """
    原始函数，使用增强实现
    为了保持向后兼容性而保留的函数签名
    """
    return enhanced_self_knowledge_retrieval_reranking(graph, question)

def cosine_similarity_manual(x, y):
    """
    手动计算余弦相似度
    输入: x, y - 两个向量或向量矩阵
    返回: 余弦相似度矩阵
    """
    # 计算点积
    dot_product = np.dot(x, y.T)
    # 计算向量的L2范数
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    # 计算余弦相似度
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim

def enhanced_is_unable_to_answer(response):
    """
    增强的响应质量验证
    检查模型响应是否表示无法回答问题
    输入: response - 模型响应文本
    返回: 是否无法回答
    """
    # 基本检查：空或太短的响应
    if not response or len(response.strip()) < 5:
        return True
    
    # 检查常见的无法回答的模式
    failure_patterns = [
        "i don't know", "cannot answer", "insufficient information",
        "unable to determine", "not enough context", "cannot provide"
    ]
    
    response_lower = response.lower()
    for pattern in failure_patterns:
        if pattern in response_lower:
            return True
    
    # 使用GPT评估响应质量（可选的高级检查）
    try:
        analysis = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": response}
            ],
            max_tokens=1,        # 只需要一个token的评分
            temperature=0.0,     # 确定性输出
            n=1,                 # 单个响应
            stop=None,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )
        score = analysis.choices[0].message.content.strip().replace("'", "").replace(".", "")
        if not score.isdigit():   
            return True
        threshold = 0.6
        if float(score) > threshold:
            return False
        else:
            return True
    except:
        return False  # 如果评估失败，假设响应是有效的

def is_unable_to_answer(response):
    """
    原始函数，使用增强实现
    为了保持向后兼容性而保留的函数签名
    """
    return enhanced_is_unable_to_answer(response)

def autowrap_text(text, font, max_width):
    """
    自动换行文本函数
    将长文本按指定宽度自动换行
    输入: text - 文本, font - 字体对象, max_width - 最大宽度
    返回: 换行后的文本行列表
    """
    text_lines = []
    
    # 如果文本宽度不超过最大宽度，直接返回
    if font.getsize(text)[0] <= max_width:
        text_lines.append(text)
    else:
        # 按单词分割文本
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            # 尽可能多地添加单词到当前行
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + ' '
                i += 1
            # 如果行为空，说明单个单词就超过了最大宽度
            if not line:
                line = words[i]
                i += 1
            text_lines.append(line)
    return text_lines

def enhanced_final_answer(question_text, response_of_KG_list_path, response_of_KG_neighbor):
    """
    增强的最终答案生成，移除置信度计算，直接使用投票机制
    结合多种证据生成最终答案
    输入: question_text - 问题文本, response_of_KG_list_path - 路径证据, 
         response_of_KG_neighbor - 邻居证据
    返回: 最终答案
    """
    # 检查是否启用增强答案生成
    if not ABLATION_CONFIG['USE_ENHANCED_ANSWER_GEN']:
        logger.info("🔬 Using basic final answer generation")
        return basic_final_answer(question_text, response_of_KG_list_path, response_of_KG_neighbor)
    
    # 处理空输入
    if response_of_KG_list_path == []:
        response_of_KG_list_path = ''
    if response_of_KG_neighbor == []:
        response_of_KG_neighbor = ''
    
    # 保留：问题类型识别和否定词处理
    question_types = identify_question_type(question_text)
    has_neg = has_negation(question_text)
    
    # 保留：KG引导推理
    try:
        kg_subgraph = []  # 构建知识图谱子图
        
        # 从路径证据中提取三元组
        if response_of_KG_list_path:
            path_lines = response_of_KG_list_path.split('\n')
            for line in path_lines:
                if '->' in line:
                    parts = line.split('->')
                    if len(parts) >= 3:
                        kg_subgraph.append([parts[0].strip(), parts[1].strip(), parts[2].strip()])
        
        # 从邻居证据中提取三元组
        if response_of_KG_neighbor:
            neighbor_lines = response_of_KG_neighbor.split('\n')
            for line in neighbor_lines:
                if '->' in line:
                    parts = line.split('->')
                    if len(parts) >= 3:
                        kg_subgraph.append([parts[0].strip(), parts[1].strip(), parts[2].strip()])
        
        # 执行KG引导推理
        if kg_subgraph and medical_reasoning_rules.kg_guided_reasoning:
            kg_guided_result = medical_reasoning_rules.kg_guided_reasoning.kg_guided_reasoning(
                question_text, kg_subgraph
            )
            logger.debug(f"KG-guided reasoning result: {kg_guided_result[:200]}...")
        
    except Exception as e:
        logger.error(f"Error in KG-guided reasoning: {e}")
    
    # 保留：根据问题类型调整推理指令
    if has_neg or 'exception' in question_types:
        reasoning_instruction = "Pay special attention to negation words and identify what should be EXCLUDED or what is NOT associated with the topic."
    else:
        reasoning_instruction = "Focus on positive associations and direct relationships."
    
    # 保留：思维链生成
    messages = [
        SystemMessage(content="You are an excellent AI assistant specialized in medical question answering with access to UMLS standardized medical knowledge and hierarchical reasoning capabilities"),
        HumanMessage(content=f'Question: {question_text}'),
        AIMessage(content=f"You have some medical knowledge information in the following:\n\n" + 
                 f'###Path-based Evidence: {response_of_KG_list_path}\n\n' + 
                 f'###Neighbor-based Evidence: {response_of_KG_neighbor}'),
        HumanMessage(content=f"Answer: Let's think step by step using hierarchical medical reasoning. {reasoning_instruction} ")
    ]
    
    # 生成思维链推理
    output_CoT = ""
    for retry in range(3):
        try:
            result_CoT = chat(messages)
            if result_CoT.content is not None and len(result_CoT.content.strip()) > 10:
                output_CoT = result_CoT.content
                break
            else:
                logger.warning(f"CoT generation attempt {retry + 1} returned insufficient content")
                sleep(5)
        except Exception as e:
            logger.error(f"CoT generation attempt {retry + 1} failed: {e}")
            sleep(5)
    
    if not output_CoT:
        logger.warning("CoT generation failed, using default reasoning")
        output_CoT = f"Based on the provided medical knowledge, I need to analyze the evidence carefully."
    
    # 保留：三次答案生成（用不同提示词）
    answers = []
    for attempt in range(3):
        try:
            # 不同的最终提示词
            final_prompts = [
                "The final answer (output the letter option) is:",
                "Based on the hierarchical analysis above, the correct answer is:",
                "Therefore, using multi-strategy reasoning, the answer choice is:"
            ]
            
            # 构建最终答案生成的消息
            messages = [
                SystemMessage(content="You are an excellent AI assistant specialized in medical question answering with access to UMLS standardized medical knowledge and hierarchical reasoning capabilities"),
                HumanMessage(content=f'Question: {question_text}'),
                AIMessage(content=f"Medical knowledge:\n\n" + 
                         f'###Path-based Evidence: {response_of_KG_list_path}\n\n' + 
                         f'###Neighbor-based Evidence: {response_of_KG_neighbor}'),
                AIMessage(content=f"Analysis: {output_CoT}"),
                AIMessage(content=final_prompts[attempt % len(final_prompts)])
            ]
            
            # 调用聊天模型生成答案
            result = chat(messages)
            if result.content is not None and len(result.content.strip()) > 0:
                # 提取答案选项（A-E）
                answer_match = re.search(r'\b([A-E])\b', result.content)
                if answer_match:
                    answers.append(answer_match.group(1))
                else:
                    # 如果没有找到选项，取前10个字符
                    answers.append(result.content.strip()[:10])
                    
        except Exception as e:
            logger.error(f"Final answer attempt {attempt + 1} failed: {e}")
            sleep(3)
    
    # 简化：直接投票选择，移除置信度计算
    if answers:
        answer_counts = Counter(answers)  # 统计每个答案的出现次数
        most_common_answer = answer_counts.most_common(1)[0][0]  # 选择出现最多的答案
        
        logger.info(f"Voting results: {dict(answer_counts)}, Selected: {most_common_answer}")
        return most_common_answer
    
    logger.error("All final answer attempts failed")
    return "A"  # 默认返回A选项

def basic_final_answer(question_text, response_of_KG_list_path, response_of_KG_neighbor):
    """
    基础版本的最终答案生成
    用于消融实验的简化版本
    输入: question_text - 问题文本, response_of_KG_list_path - 路径证据, 
         response_of_KG_neighbor - 邻居证据
    返回: 最终答案
    """
    # 处理空输入
    if response_of_KG_list_path == []:
        response_of_KG_list_path = ''
    if response_of_KG_neighbor == []:
        response_of_KG_neighbor = ''
    
    # 简单的消息构建
    messages = [
        SystemMessage(content="You are a medical AI assistant."),
        HumanMessage(content=f'Question: {question_text}'),
        AIMessage(content=f"Knowledge:\n{response_of_KG_list_path}\n{response_of_KG_neighbor}"),
        HumanMessage(content="Answer: The final answer is:")
    ]
    
    try:
        result = chat(messages)
        # 提取答案选项
        answer_match = re.search(r'\b([A-E])\b', result.content)
        return answer_match.group(1) if answer_match else "A"
    except:
        return "A"  # 默认返回A

def final_answer(str, response_of_KG_list_path, response_of_KG_neighbor):
    """
    原始函数签名保持不变
    为了保持向后兼容性而保留的函数包装器
    """
    return enhanced_final_answer(str, response_of_KG_list_path, response_of_KG_neighbor)

@retry_on_failure()
def prompt_document(question,instruction):
    """
    原始文档提示模板
    基于医学知识回答患者问题
    输入: question - 患者问题, instruction - 医学知识指导
    返回: 医学建议
    """
    # 医学咨询的提示模板
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

    # 调用聊天模型
    response_document_bm25 = chat(chat_prompt_with_values.to_messages()).content
    return response_document_bm25

def load_and_clean_triples(file_path):
    """
    从CSV文件加载和清理知识图谱三元组
    处理原始数据并过滤无效三元组
    输入: file_path - 文件路径
    返回: 清理后的DataFrame
    """
    logger.info("Loading knowledge graph triples...")
    
    # 加载CSV文件（制表符分隔，无标题行）
    df = pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    df_clean = df.dropna().copy()  # 去除包含NaN的行
    
    # 清理数据：转换为字符串并去除空白
    df_clean.loc[:, 'head'] = df_clean['head'].astype(str).str.strip()
    df_clean.loc[:, 'relation'] = df_clean['relation'].astype(str).str.strip()
    df_clean.loc[:, 'tail'] = df_clean['tail'].astype(str).str.strip()
    
    # 过滤空字符串
    df_clean = df_clean[(df_clean['head'] != '') & 
                       (df_clean['relation'] != '') & 
                       (df_clean['tail'] != '')]
    
    logger.info(f"Loaded {len(df)} total triples, {len(df_clean)} valid triples after cleaning")
    
    return df_clean

# ========================= 主程序执行部分 =========================
if __name__ == "__main__":
    # 配置第三方API
    openai.api_key = "sk-P4hNAfoKF4JLckjCuE99XbaN4bZIORZDPllgpwh6PnYWv4cj"  # OpenAI API密钥
    openai.api_base = "https://aiyjg.lol/v1"  # API基础URL
    
    # 设置环境变量
    os.environ['OPENAI_API_KEY'] = openai.api_key

    # 1. 构建neo4j知识图谱数据集
    uri = "bolt://localhost:7688"  # Neo4j数据库连接URI
    username = "neo4j"             # 用户名
    password = "Cyber@511"         # 密码

    # 创建Neo4j驱动和会话
    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()

    # 清理现有知识图谱
    logger.info("Cleaning existing knowledge graph...")
    session.run("MATCH (n) DETACH DELETE n")  # 删除所有节点和关系

    # 加载和清理三元组数据
    df_clean = load_and_clean_triples('./Alzheimers/train_s2s.txt')

    # 批量插入配置
    batch_size = 1000      # 批次大小
    valid_triples = 0      # 有效三元组计数
    batch_queries = []     # 批次查询列表
    batch_params = []      # 批次参数列表
    
    logger.info("Starting batch insertion of knowledge graph triples...")
    
    # 遍历清理后的数据进行批量插入
    for index, row in tqdm(df_clean.iterrows(), desc="Building knowledge graph"):
        head_name = row['head']      # 头实体
        tail_name = row['tail']      # 尾实体
        relation_name = row['relation']  # 关系
        
        # 验证三元组质量
        if not validate_knowledge_triple(head_name, relation_name, tail_name):
            continue

        # 构建Cypher查询（MERGE确保节点和关系的唯一性）
        query = (
            "MERGE (h:Entity { name: $head_name }) "        # 创建或匹配头实体
            "MERGE (t:Entity { name: $tail_name }) "        # 创建或匹配尾实体
            "MERGE (h)-[r:`" + relation_name + "`]->(t)"    # 创建或匹配关系
        )
        
        # 添加到批次
        batch_queries.append(query)
        batch_params.append({
            'head_name': head_name,
            'tail_name': tail_name,
            'relation_name': relation_name
        })
        
        # 当达到批次大小时执行批次插入
        if len(batch_queries) >= batch_size:
            try:
                # 使用事务执行批次
                with driver.session() as batch_session:
                    tx = batch_session.begin_transaction()
                    for q, params in zip(batch_queries, batch_params):
                        tx.run(q, **params)
                    tx.commit()  # 提交事务
                valid_triples += len(batch_queries)
                logger.debug(f"Successfully inserted batch of {len(batch_queries)} triples")
            except Exception as e:
                logger.error(f"Failed to insert batch: {e}")
                # 如果批次失败，尝试单个插入
                for q, params in zip(batch_queries, batch_params):
                    try:
                        session.run(q, **params)
                        valid_triples += 1
                    except Exception as single_e:
                        logger.warning(f"Failed to insert single triple: {params['head_name']} -> {params['relation_name']} -> {params['tail_name']}, Error: {single_e}")
            
            # 重置批次
            batch_queries = []
            batch_params = []
    
    # 处理最后一个不完整的批次
    if batch_queries:
        try:
            with driver.session() as batch_session:
                tx = batch_session.begin_transaction()
                for q, params in zip(batch_queries, batch_params):
                    tx.run(q, **params)
                tx.commit()
            valid_triples += len(batch_queries)
            logger.debug(f"Successfully inserted final batch of {len(batch_queries)} triples")
        except Exception as e:
            logger.error(f"Failed to insert final batch: {e}")
            for q, params in zip(batch_queries, batch_params):
                try:
                    session.run(q, **params)
                    valid_triples += 1
                except Exception as single_e:
                    logger.warning(f"Failed to insert single triple: {params['head_name']} -> {params['relation_name']} -> {params['tail_name']}, Error: {single_e}")

    logger.info(f"Successfully inserted {valid_triples} valid triples using batch processing")

    # 构建层次化知识图谱结构
    logger.info("Building hierarchical knowledge graph structure...")
    flat_kg_triples = []
    for _, row in df_clean.iterrows():
        flat_kg_triples.append([row['head'], row['relation'], row['tail']])
    
    # 构建层次化结构
    hierarchical_kg_framework.build_hierarchical_structure(flat_kg_triples)

    # 2. 初始化OpenAI API客户端用于后续推理
    OPENAI_API_KEY = openai.api_key
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo', temperature=0.7)

    # 加载预训练的嵌入
    logger.info("Loading embeddings...")
    with open('./Alzheimers/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)  # 实体嵌入
        
    with open('./Alzheimers/keyword_embeddings.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)  # 关键词嵌入

    # 初始化医学推理规则的KG引导推理
    medical_reasoning_rules.initialize_kg_guided_reasoning(flat_kg_triples, chat)

    # 3. 处理各个数据集
    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset}")
        processor = dataset2processor[dataset]()  # 获取数据集处理器
        data = processor.load_dataset()           # 加载数据集

        acc, total_num = 0, 0      # 准确率统计
        generated_data = []        # 生成的数据列表

        # 遍历数据集中的每个项目
        for item in tqdm(data, desc=f"Processing {dataset}"):
            
            # 定期清理资源
            if total_num > 0 and total_num % CLEANUP_FREQUENCY == 0:
                cleanup_resources(total_num)
            
            # 生成输入文本和提取实体
            input_text = [processor.generate_prompt(item)]
            entity_list = item['entity'].split('\n')
            question_kg = []
            
            # 处理实体列表
            for entity in entity_list:
                try:
                    entity = entity.split('.')[1].strip()  # 去除编号
                    question_kg.append(entity)
                except:
                    continue

            # 识别问题类型
            question_types = identify_question_type(input_text[0])
            logger.info(f"Question types identified: {question_types}")

            # 执行增强的实体匹配
            match_kg, confidence_scores = enhanced_entity_matching(
                question_kg, entity_embeddings, keyword_embeddings, input_text[0])

            # 确保至少有两个实体用于路径查找
            if len(match_kg) < 2:
                logger.warning(f"Insufficient entities matched for question: {input_text[0][:100]}...")
                match_kg.extend(question_kg[:2])

            # 4. 增强的neo4j知识图谱路径查找
            if len(match_kg) > 1:
                start_entity = match_kg[0]      # 起始实体
                candidate_entity = match_kg[1:] # 候选实体列表
                
                result_path_list = []  # 结果路径列表
                
                # 复杂的路径查找逻辑
                while True:
                    flag = 0           # 标志变量
                    paths_list = []    # 路径列表
                    
                    # 遍历候选实体
                    while candidate_entity:
                        end_entity = candidate_entity[0]
                        candidate_entity.remove(end_entity)
                        
                        # 查找最短路径
                        paths, exist_entity = find_shortest_path(start_entity, end_entity, candidate_entity, question_types)
                        path_list = []
                        
                        # 处理路径结果
                        if paths == [''] or paths == []:
                            flag = 1
                            if not candidate_entity:
                                flag = 0
                                break
                            start_entity = candidate_entity[0]
                            candidate_entity.remove(start_entity)
                            break
                        else:
                            # 分割路径
                            for p in paths:
                                path_list.append(p.split('->'))
                            if path_list:
                                paths_list.append(path_list)
                        
                        # 处理存在的实体
                        if exist_entity != {}:
                            try:
                                candidate_entity.remove(exist_entity)
                            except:
                                continue
                        start_entity = end_entity
                    
                    # 组合路径
                    result_path = combine_lists(*paths_list)
                    
                    if result_path:
                        result_path_list.extend(result_path)
                    if flag == 1:
                        continue
                    else:
                        break
                
                # 处理路径结果
                start_tmp = []
                for path_new in result_path_list:
                    if path_new == []:
                        continue
                    if path_new[0] not in start_tmp:
                        start_tmp.append(path_new[0])
                
                # 根据起始实体数量选择路径
                if len(start_tmp) == 0:
                    result_path = {}
                    single_path = {}
                else:
                    if len(start_tmp) == 1:
                        result_path = result_path_list[:5]
                    else:
                        result_path = []
                        
                        # 复杂的路径选择逻辑
                        if len(start_tmp) >= 5:
                            for path_new in result_path_list:
                                if path_new == []:
                                    continue
                                if path_new[0] in start_tmp:
                                    result_path.append(path_new)
                                    start_tmp.remove(path_new[0])
                                if len(result_path) == 5:
                                    break
                        else:
                            count = 5 // len(start_tmp)
                            remind = 5 % len(start_tmp)
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

                                        if len(start_tmp) == 1:
                                            count = count + remind
                                else:
                                    break

                    try:
                        single_path = result_path_list[0]
                    except:
                        single_path = result_path_list
                    
            else:
                result_path = {}
                single_path = {}            
            
            # 5. 增强的neo4j知识图谱邻居实体
            neighbor_list = []         # 邻居列表
            neighbor_list_disease = [] # 疾病邻居列表
            
            # 为每个匹配的实体获取邻居
            for match_entity in match_kg:
                disease_flag = 0
                neighbors, disease = get_entity_neighbors(match_entity, disease_flag, question_types)
                neighbor_list.extend(neighbors)

                # 添加层次化上下文
                try:
                    hierarchical_context = hierarchical_kg_framework.get_hierarchical_context(
                        match_entity, context_type='all'
                    )
                    
                    # 将层次化上下文转换为邻居格式
                    for context_type, context_items in hierarchical_context.items():
                        for context_item in context_items[:3]:
                            if isinstance(context_item, dict):
                                entity_name = context_item.get('entity', '')
                                relation_name = context_item.get('relation', '')
                                if entity_name and relation_name:
                                    neighbor_entry = [
                                        match_entity.replace("_", " "),
                                        f"hierarchical_{relation_name}".replace("_", " "),
                                        entity_name.replace("_", " ")
                                    ]
                                    neighbor_list.append(neighbor_entry)
                                    
                except Exception as e:
                    logger.error(f"Error getting hierarchical context for {match_entity}: {e}")

                # 处理疾病相关邻居
                while disease:
                    new_disease = []
                    for disease_tmp in disease:
                        if disease_tmp in match_kg:
                            new_disease.append(disease_tmp)

                    if new_disease:
                        for disease_entity in new_disease:
                            disease_flag = 1
                            neighbors, disease = get_entity_neighbors(disease_entity, disease_flag, question_types)
                            neighbor_list_disease.extend(neighbors)
                    else:
                        for disease_entity in disease:
                            disease_flag = 1
                            neighbors, disease = get_entity_neighbors(disease_entity, disease_flag, question_types)
                            neighbor_list_disease.extend(neighbors)
                    
                    if len(neighbor_list_disease) > 10:
                        break
            
            # 如果邻居不够，添加疾病邻居
            if len(neighbor_list) <= 5:
                neighbor_list.extend(neighbor_list_disease)

            # 6. 增强的知识图谱路径基础提示生成
            if len(match_kg) > 1:
                response_of_KG_list_path = []
                if result_path == {}:
                    response_of_KG_list_path = []
                    path_sampled = []
                else:
                    # 格式化路径
                    result_new_path = []
                    for total_path_i in result_path:
                        path_input = "->".join(total_path_i)
                        result_new_path.append(path_input)
                    
                    # 重排序和生成自然语言
                    path = "\n".join(result_new_path)
                    path_sampled = self_knowledge_retrieval_reranking(path, input_text[0])
                    response_of_KG_list_path = prompt_path_finding(path_sampled)
            else:
                response_of_KG_list_path = '{}'

            # 处理单个路径
            try:
                response_single_path = prompt_path_finding(single_path)
                if is_unable_to_answer(response_single_path):
                    response_single_path = prompt_path_finding(single_path)
            except:
                response_single_path = ""

            # 7. 增强的知识图谱邻居实体基础提示生成
            response_of_KG_list_neighbor = []
            neighbor_new_list = []
            
            # 格式化邻居信息
            for neighbor_i in neighbor_list:
                neighbor = "->".join(neighbor_i)
                neighbor_new_list.append(neighbor)

            # 选择前5个邻居
            if len(neighbor_new_list) > 5:
                neighbor_input = "\n".join(neighbor_new_list[:5])
            else:
                neighbor_input = "\n".join(neighbor_new_list)
            
            # 重排序和生成自然语言
            neighbor_input_sampled = self_knowledge_retrieval_reranking(neighbor_input, input_text[0])
            response_of_KG_neighbor = prompt_neighbor(neighbor_input_sampled)

            # 8. 增强的基于提示的医学对话答案生成（移除了置信度计算）
            output_all = enhanced_final_answer(input_text[0], response_of_KG_list_path, response_of_KG_neighbor)

            # 解析结果并统计准确率
            ret_parsed, acc_item = processor.parse(output_all, item)
            ret_parsed['path'] = path_sampled if 'path_sampled' in locals() else ""
            ret_parsed['neighbor_input'] = neighbor_input_sampled if 'neighbor_input_sampled' in locals() else ""
            ret_parsed['response_of_KG_list_path'] = response_of_KG_list_path
            ret_parsed['response_of_KG_neighbor'] = response_of_KG_neighbor
            ret_parsed['entity_confidence_scores'] = confidence_scores if 'confidence_scores' in locals() else []
            ret_parsed['question_types'] = question_types
            
            # 添加增强处理的结果
            try:
                # UMLS标准化实体
                ret_parsed['umls_normalized_entities'] = umls_normalizer.normalize_medical_terms(question_kg)
                ret_parsed['umls_semantic_variants'] = [umls_normalizer.get_semantic_variants(entity)[:3] for entity in question_kg[:3]]
                
                # 增强实体链接
                enhanced_links = umls_normalizer.enhanced_entity_linking_method(
                    question_kg, input_text[0], question_types
                )
                ret_parsed['enhanced_entity_links'] = enhanced_links
                
                # 自适应知识选择
                adaptive_knowledge = umls_normalizer.adaptive_knowledge_selection(
                    question_types, question_kg
                )
                ret_parsed['adaptive_knowledge_count'] = len(adaptive_knowledge)
                
                # 层次化上下文
                hierarchical_contexts = {}
                for entity in question_kg[:3]:
                    hierarchical_contexts[entity] = hierarchical_kg_framework.get_hierarchical_context(
                        entity, context_type='all'
                    )
                ret_parsed['hierarchical_contexts'] = hierarchical_contexts
                
                # 多跳路径
                if len(question_kg) >= 2:
                    multi_hop_paths = multi_hop_reasoner.optimized_multi_hop.intelligent_path_selection(
                        question_kg[:1], question_kg[1:2], max_hops=2
                    )
                    ret_parsed['multi_hop_paths_count'] = len(multi_hop_paths)
                else:
                    ret_parsed['multi_hop_paths_count'] = 0
                
            except Exception as e:
                logger.error(f"Error in enhanced processing: {e}")
                # 设置默认值
                ret_parsed['umls_normalized_entities'] = question_kg
                ret_parsed['umls_semantic_variants'] = []
                ret_parsed['enhanced_entity_links'] = {}
                ret_parsed['adaptive_knowledge_count'] = 0
                ret_parsed['hierarchical_contexts'] = {}
                ret_parsed['multi_hop_paths_count'] = 0
            
            # 转换NumPy类型为Python原生类型
            ret_parsed = convert_numpy_types(ret_parsed)
            
            # 统计准确率
            if ret_parsed['prediction'] in processor.num2answer.values():
                acc += acc_item
                total_num += 1
            generated_data.append(ret_parsed)

        # 输出数据集处理结果
        logger.info(f"Dataset: {dataset}")
        logger.info(f"Accuracy: {acc/total_num:.4f} ({acc}/{total_num})")

        # 创建输出目录
        os.makedirs('./Alzheimers/result_chatgpt_mindmap', exist_ok=True)
        
        # 保存消融实验结果
        output_filename = f"{dataset}_{CURRENT_ABLATION_CONFIG}_ablation_results.json"
        with open(os.path.join('./Alzheimers/result_chatgpt_mindmap', output_filename), 'w') as f:
            json.dump(generated_data, fp=f, indent=2)
            
        logger.info(f"Ablation results saved for dataset: {dataset}")
        
        # 生成性能统计
        performance_stats = {
            'ablation_config': CURRENT_ABLATION_CONFIG,
            'config_details': ABLATION_CONFIG,
            'dataset': dataset,
            'total_questions': total_num,
            'correct_answers': acc,
            'accuracy': acc/total_num if total_num > 0 else 0,
            'avg_entity_confidence': np.mean([
                np.mean(item.get('entity_confidence_scores', [0])) 
                for item in generated_data if item.get('entity_confidence_scores')
            ]) if generated_data else 0,
            'question_type_distribution': {},
            'hierarchical_context_coverage': 0,
            'multi_strategy_usage': 0
        }
        
        # 统计问题类型分布
        question_type_counts = {}
        hierarchical_coverage_count = 0
        multi_strategy_count = 0
        
        for item in generated_data:
            q_types = item.get('question_types', [])
            for q_type in q_types:
                question_type_counts[q_type] = question_type_counts.get(q_type, 0) + 1
            
            if item.get('hierarchical_contexts'):
                hierarchical_coverage_count += 1
            
            if item.get('enhanced_entity_links'):
                multi_strategy_count += 1
        
        # 完善统计信息
        performance_stats['question_type_distribution'] = question_type_counts
        performance_stats['hierarchical_context_coverage'] = hierarchical_coverage_count / len(generated_data) if generated_data else 0
        performance_stats['multi_strategy_usage'] = multi_strategy_count / len(generated_data) if generated_data else 0
        
        # 保存性能统计
        stats_filename = f"{dataset}_{CURRENT_ABLATION_CONFIG}_performance_stats.json"
        with open(os.path.join('./Alzheimers/result_chatgpt_mindmap', stats_filename), 'w') as f:
            json.dump(performance_stats, fp=f, indent=2)
            
        # 记录统计结果
        logger.info(f"Performance statistics saved for dataset: {dataset}")
        logger.info(f"Hierarchical context coverage: {performance_stats['hierarchical_context_coverage']:.3f}")
        logger.info(f"Multi-strategy usage: {performance_stats['multi_strategy_usage']:.3f}")

    # 输出消融实验完成信息
    logger.info("="*50)
    logger.info(f"🎉 Ablation study completed for configuration: {CURRENT_ABLATION_CONFIG}")
    logger.info("📊 Ablation configuration applied:")
    for module, enabled in ABLATION_CONFIG.items():
        status = "✅ ENABLED" if enabled else "❌ DISABLED"
        logger.info(f"   {module}: {status}")
    logger.info("="*50)
    
    # 生成总体统计报告
    overall_stats = {
        'current_ablation_config': CURRENT_ABLATION_CONFIG,
        'ablation_configuration': ABLATION_CONFIG,
        'available_configs': list(ABLATION_CONFIGS.keys()),
        'performance_config': {
            'cleanup_frequency': CLEANUP_FREQUENCY,
            'max_cache_size': MAX_CACHE_SIZE,
            'keep_cache_size': KEEP_CACHE_SIZE,
            'max_failed_cuis': MAX_FAILED_CUIS
        },
        'datasets_processed': datasets,
        'total_datasets': len(datasets),
        'processing_timestamp': datetime.now().isoformat()
    }
    
    # 保存实验报告
    with open('./Alzheimers/result_chatgpt_mindmap/ablation_experiment_report.json', 'w') as f:
        json.dump(overall_stats, fp=f, indent=2)
    
    logger.info("📈 Ablation experiment report saved!")
    
    # 关闭数据库连接
    driver.close()
    
    # 输出完成信息
    logger.info("🔌 Database connection closed. Ablation study complete!")
    logger.info(f"🔬 To run different ablation configurations, set ABLATION_CONFIG environment variable to one of: {list(ABLATION_CONFIGS.keys())}")

# ==============================================================================
# 代码结束
# ==============================================================================