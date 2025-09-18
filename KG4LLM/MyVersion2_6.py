from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate,LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import numpy as np
import re
import string
from neo4j import GraphDatabase, basic_auth
import pandas as pd
from collections import deque, Counter, defaultdict
import itertools
from typing import Dict, List, Tuple, Optional
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize 
import openai
from langchain.llms import OpenAI
import os
from time import sleep
import logging
from functools import wraps
from datetime import datetime
import gc
import requests

from dataset_utils import *
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================= 统一阈值配置管理 =========================


class ThresholdConfig:
    """统一管理所有语义相似度阈值的配置类"""
    
    def __init__(self, config_name='default'):
        self.config_name = config_name
        self._load_config(config_name)
    
    def _load_config(self, config_name):
        """加载指定配置"""
        configs = {
            'default': {
                # 实体匹配相关阈值，语义相似度
                'entity_matching': {
                    'basic_similarity': 0.6,           # 基础实体匹配阈值
                    'enhanced_similarity': 0.6,        # 增强实体匹配阈值
                    'confidence_threshold': 0.85,      # 实体置信度阈值
                    'min_similarity': 0.6,             # 最小相似度阈值
                    'negation_factor': 0.8,            # 否定问题的阈值调整因子
                },
                
                # 语义匹配相关阈值，语义相似度
                'semantic_matching': {
                    'jaccard_similarity': 0.7,         # SemanticMatcher的Jaccard相似度阈值
                    'vector_similarity': 0.7,          # 向量相似度阈值
                    'keyword_matching': 0.3,           # 关键词匹配阈值
                },
                
                # 问题分类相关阈值
                'question_classification': {
                    'type_similarity': 0.4,            # 问题类型分类阈值，语义相似度
                    'secondary_threshold': 0.85,       # 第二相似类型的阈值因子
                },
                
                # 医学概念分类阈值，语义相似度
                'medical_concept': {
                    'disease': 0.65,                   # 疾病概念阈值
                    'symptom': 0.60,                   # 症状概念阈值
                    'treatment': 0.58,                 # 治疗概念阈值
                    'general': 0.55,                   # 通用医学概念阈值
                },
                
                # 层次化图谱相关阈值
                'hierarchical_kg': {
                    'semantic_matching': 0.7,          # 层次化图谱语义匹配阈值，语义相似度
                    'concept_center': 0.65,            # 概念中心相似度阈值，语义相似度
                    'hierarchy_weight': 0.75,          # 层次关系权重阈值
                },
                
                # 知识质量评估阈值
                'knowledge_quality': {
                    'quality_threshold': 0.7,          # 知识质量阈值
                    'relation_importance': 0.5,        # 关系重要性阈值
                    'path_confidence': 0.1,            # 路径置信度阈值
                },
                
                # 多跳推理相关阈值
                'multi_hop': {
                    'path_quality': 0.5,               # 路径质量阈值
                    'reasoning_confidence': 0.6,       # 推理置信度阈值
                    'evidence_weight': 0.4,            # 证据权重阈值
                }
            },
            
            'my_settings': {
                # 实体匹配相关阈值，语义相似度
                'entity_matching': {
                    'basic_similarity': 0.8,           # 基础实体匹配阈值
                    'enhanced_similarity': 0.8,        # 增强实体匹配阈值
                    'confidence_threshold': 0.9,      # 实体置信度阈值
                    'min_similarity': 0.7,             # 最小相似度阈值
                    'negation_factor': 0.85,            # 否定问题的阈值调整因子
                },
                
                # 语义匹配相关阈值，语义相似度
                'semantic_matching': {
                    'jaccard_similarity': 0.8,         # SemanticMatcher的Jaccard相似度阈值
                    'vector_similarity': 0.8,          # 向量相似度阈值
                    'keyword_matching': 0.4,           # 关键词匹配阈值
                },
                
                # 问题分类相关阈值
                'question_classification': {
                    'type_similarity': 0.4,            # 问题类型分类阈值，语义相似度
                    'secondary_threshold': 0.85,       # 第二相似类型的阈值因子
                },
                
                # 医学概念分类阈值，语义相似度
                'medical_concept': {
                    'disease': 0.8,                   # 疾病概念阈值
                    'symptom': 0.8,                   # 症状概念阈值
                    'treatment': 0.8,                 # 治疗概念阈值
                    'general': 0.8,                   # 通用医学概念阈值
                },
                
                # 层次化图谱相关阈值
                'hierarchical_kg': {
                    'semantic_matching': 0.8,          # 层次化图谱语义匹配阈值，语义相似度
                    'concept_center': 0.8,            # 概念中心相似度阈值，语义相似度
                    'hierarchy_weight': 0.75,          # 层次关系权重阈值
                },
                
                # 知识质量评估阈值
                'knowledge_quality': {
                    'quality_threshold': 0.7,          # 知识质量阈值
                    'relation_importance': 0.5,        # 关系重要性阈值
                    'path_confidence': 0.1,            # 路径置信度阈值
                },
                
                # 多跳推理相关阈值
                'multi_hop': {
                    'path_quality': 0.5,               # 路径质量阈值
                    'reasoning_confidence': 0.6,       # 推理置信度阈值
                    'evidence_weight': 0.4,            # 证据权重阈值
                }
            },
            
            'strict': {
                # 严格模式：更高的阈值
                'entity_matching': {
                    'basic_similarity': 0.75,
                    'enhanced_similarity': 0.75,
                    'confidence_threshold': 0.9,
                    'min_similarity': 0.7,
                    'negation_factor': 0.85,
                },
                'semantic_matching': {
                    'jaccard_similarity': 0.8,
                    'vector_similarity': 0.8,
                    'keyword_matching': 0.4,
                },
                'question_classification': {
                    'type_similarity': 0.5,
                    'secondary_threshold': 0.9,
                },
                'medical_concept': {
                    'disease': 0.75,
                    'symptom': 0.7,
                    'treatment': 0.68,
                    'general': 0.65,
                },
                'hierarchical_kg': {
                    'semantic_matching': 0.8,
                    'concept_center': 0.75,
                    'hierarchy_weight': 0.8,
                },
                'knowledge_quality': {
                    'quality_threshold': 0.8,
                    'relation_importance': 0.6,
                    'path_confidence': 0.2,
                },
                'multi_hop': {
                    'path_quality': 0.6,
                    'reasoning_confidence': 0.7,
                    'evidence_weight': 0.5,
                }
            }
        }

        if config_name not in configs:
            logger.warning(f"Unknown threshold config '{config_name}', using 'default'")
            config_name = 'default'
        
        # 加载配置
        config = configs[config_name]
        for category, thresholds in config.items():
            setattr(self, category, thresholds)
        
        logger.info(f"Loaded threshold configuration: {config_name}")
    
    def get_threshold(self, category, key):
        """获取指定类别和键的阈值"""
        try:
            category_config = getattr(self, category)
            return category_config.get(key, 0.5)  # 默认返回0.5
        except AttributeError:
            logger.warning(f"Unknown threshold category: {category}")
            return 0.5
    
    def set_threshold(self, category, key, value):
        """动态设置阈值"""
        try:
            category_config = getattr(self, category)
            category_config[key] = value
            logger.info(f"Updated threshold {category}.{key} = {value}")
        except AttributeError:
            logger.warning(f"Cannot set threshold for unknown category: {category}")
    
    def get_concept_threshold(self, concept_type):
        """根据概念类型获取对应阈值"""
        concept_type = concept_type.lower()
        if concept_type in self.medical_concept:
            return self.medical_concept[concept_type]
        else:
            return self.medical_concept['general']
    
    def adjust_for_negation(self, base_threshold):
        """为否定问题调整阈值"""
        return base_threshold * self.entity_matching['negation_factor']
    
    def print_config(self):
        """打印当前配置"""
        print(f"\n=== Threshold Configuration: {self.config_name} ===")
        for category in ['entity_matching', 'semantic_matching', 'question_classification', 
                        'medical_concept', 'hierarchical_kg', 'knowledge_quality', 'multi_hop']:
            if hasattr(self, category):
                print(f"\n{category}:")
                config = getattr(self, category)
                for key, value in config.items():
                    print(f"  {key}: {value}")

    
threshold_mode = os.getenv('THRESHOLD_MODE', 'default')
THRESHOLDS = ThresholdConfig(threshold_mode)


# ========================= 消融实验配置 =========================
# 🔬 消融实验开关配置
ABLATION_CONFIGS = {
    'baseline': {
        'USE_HIERARCHICAL_KG': False,
        'USE_MULTI_STRATEGY_LINKING': False,
        'USE_ADAPTIVE_UMLS': False,
        'USE_UMLS_NORMALIZATION': False,
        'USE_REASONING_RULES': False,
        'USE_KG_GUIDED_REASONING': False,
        'USE_OPTIMIZED_MULTIHOP': False,
        'USE_ENHANCED_ANSWER_GEN': False
    },
    'full_model': {
        'USE_HIERARCHICAL_KG': True,
        'USE_MULTI_STRATEGY_LINKING': True,
        'USE_ADAPTIVE_UMLS': True,
        'USE_UMLS_NORMALIZATION': True,
        'USE_REASONING_RULES': True,
        'USE_KG_GUIDED_REASONING': True,
        'USE_OPTIMIZED_MULTIHOP': True,
        'USE_ENHANCED_ANSWER_GEN': True
    },
    'ablation_hierarchical_kg': {
        'USE_HIERARCHICAL_KG': False,
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
        'USE_MULTI_STRATEGY_LINKING': False,
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
        'USE_ADAPTIVE_UMLS': False,
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
        'USE_UMLS_NORMALIZATION': False,
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
        'USE_REASONING_RULES': False,
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
        'USE_KG_GUIDED_REASONING': False,
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
        'USE_OPTIMIZED_MULTIHOP': False,
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
        'USE_ENHANCED_ANSWER_GEN': False
    },
    'myself_settings': {
        'USE_HIERARCHICAL_KG': True,
        'USE_MULTI_STRATEGY_LINKING': True,
        'USE_ADAPTIVE_UMLS': True,
        'USE_UMLS_NORMALIZATION': True,
        'USE_REASONING_RULES': False,
        'USE_KG_GUIDED_REASONING': False,
        'USE_OPTIMIZED_MULTIHOP': True,
        'USE_ENHANCED_ANSWER_GEN': True
    }
}

# 当前实验配置 (可以通过命令行参数或环境变量修改)
CURRENT_ABLATION_CONFIG = os.getenv('ABLATION_CONFIG', 'ablation_kg_guided')


def get_ablation_config():
    """获取当前消融实验配置"""
    config = ABLATION_CONFIGS.get(CURRENT_ABLATION_CONFIG, ABLATION_CONFIGS['full_model'])
    logger.info(f"🔬 Using ablation configuration: {CURRENT_ABLATION_CONFIG}")
    logger.info(f"📋 Configuration details: {config}")
    return config

# 获取当前配置
ABLATION_CONFIG = get_ablation_config()

# ========================= 性能优化配置 =========================
CLEANUP_FREQUENCY = 15
MAX_CACHE_SIZE = 1500
KEEP_CACHE_SIZE = 800
MAX_FAILED_CUIS = 1000

# Enhanced API retry configuration
MAX_RETRIES = 60
RETRY_WAIT_TIME = 60
ENTITY_CONFIDENCE_THRESHOLD = 0.85
KNOWLEDGE_QUALITY_THRESHOLD = 0.7
MIN_SIMILARITY_THRESHOLD = 0.6

# ========================= 语义问题分类器 =========================

class SemanticQuestionTypeClassifier:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', similarity_threshold=0.4):
        """
        初始化语义问题类型分类器
        
        Args:
            model_name: 使用的sentence transformer模型
            similarity_threshold: 相似度阈值
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Loaded semantic model: {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load semantic model: {e}")
            raise
            
        # 使用统一配置的阈值
        self.similarity_threshold = THRESHOLDS.get_threshold('question_classification', 'type_similarity')
        
        # 定义每个问题类型的典型例句
        self.type_examples = {
            'definition': [
                "What is Alzheimer's disease?",
                "Define dementia",
                "What does cognitive impairment mean?",
                "Alzheimer's disease is characterized by",
                "The meaning of neurodegeneration"
            ],
            'causation': [
                "What causes Alzheimer's disease?",
                "Why does dementia occur?",
                "What leads to memory loss?",
                "The etiology of cognitive decline",
                "Alzheimer's disease is caused by",
                "Due to what factors does dementia develop?"
            ],
            'treatment': [
                "How to treat Alzheimer's disease?",
                "What medication for dementia?",
                "Treatment options for cognitive decline",
                "Therapy for memory loss",
                "Drugs used to treat Alzheimer's",
                "Management of dementia patients"
            ],
            'symptom': [
                "What are symptoms of Alzheimer's?",
                "Signs of dementia",
                "Clinical manifestations of cognitive decline",
                "How does Alzheimer's present?",
                "Symptoms include memory loss",
                "Patient presents with confusion"
            ],
            'diagnosis': [
                "How to diagnose Alzheimer's disease?",
                "What tests for dementia?",
                "Diagnostic criteria for cognitive impairment",
                "How to identify Alzheimer's?",
                "Assessment of memory problems",
                "Evaluation of cognitive function"
            ],
            'prevention': [
                "How to prevent Alzheimer's disease?",
                "Prevention of dementia",
                "How to avoid cognitive decline?",
                "Reducing risk of memory loss",
                "Protective factors against Alzheimer's",
                "Ways to prevent neurodegeneration"
            ],
            'exception': [
                "All of the following EXCEPT",
                "Which is NOT associated with",
                "All are true EXCEPT",
                "The following are false EXCEPT",
                "Not a characteristic of",
                "All predisposing factors EXCEPT"
            ]
        }
        
        # 预计算所有例句的嵌入向量
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """预计算所有类型例句的嵌入向量"""
        self.type_embeddings = {}
        
        for q_type, examples in self.type_examples.items():
            # 计算所有例句的嵌入向量
            embeddings = self.model.encode(examples, show_progress_bar=False)  # 禁用，避免7个重复进度条
            # 取平均作为该类型的代表向量
            self.type_embeddings[q_type] = np.mean(embeddings, axis=0)
            
        logger.info(f"✅ Precomputed embeddings for {len(self.type_embeddings)} question types")
    
    def identify_question_type(self, question):
        """
        使用语义匹配识别问题类型
        
        Args:
            question: 输入问题文本
            
        Returns:
            list: 识别的问题类型列表
        """
        try:
            # 计算问题的嵌入向量
            question_embedding = self.model.encode([question], show_progress_bar=False)[0]  # 禁用进度条
            
            # 计算与各个类型的相似度
            similarities = {}
            for q_type, type_embedding in self.type_embeddings.items():
                similarity = cosine_similarity(
                    question_embedding.reshape(1, -1),
                    type_embedding.reshape(1, -1)
                )[0][0]
                similarities[q_type] = similarity
            
            # 特殊处理否定/例外问题
            if self._is_exception_question(question):
                similarities['exception'] = max(similarities.get('exception', 0), 0.8)
            
            # 选择相似度超过阈值的类型
            matched_types = []
            for q_type, similarity in similarities.items():
                if similarity >= self.similarity_threshold:
                    matched_types.append((q_type, similarity))
            
            # 按相似度排序
            matched_types.sort(key=lambda x: x[1], reverse=True)
            
            # 返回最相似的类型(们)
            if matched_types:
                result_types = [matched_types[0][0]]  # 至少返回最相似的
                
                # 如果第二相似的类型分数接近，也包含进来
                if (len(matched_types) > 1 and 
                    matched_types[1][1] >= matched_types[0][1] * 0.85):
                    result_types.append(matched_types[1][0])
                
                logger.debug(f"Question: '{question[:50]}...' -> Types: {result_types}")
                return result_types
            else:
                logger.debug(f"Question: '{question[:50]}...' -> Type: ['general'] (no match)")
                return ['general']
                
        except Exception as e:
            logger.error(f"Error in semantic question type identification: {e}")
            return ['general']
    
    def _is_exception_question(self, question):
        """检测是否为例外/否定类型问题"""
        exception_indicators = [
            'except', 'not', 'false', 'incorrect', 'exclude', 
            'excluding', 'other than', 'rather than', 'not true',
            'not associated', 'not characteristic'
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in exception_indicators)
    
    def get_similarity_scores(self, question):
        """获取问题与所有类型的相似度分数(用于调试)"""
        question_embedding = self.model.encode([question], show_progress_bar=False)[0]  # 禁用进度条
        
        similarities = {}
        for q_type, type_embedding in self.type_embeddings.items():
            similarity = cosine_similarity(
                question_embedding.reshape(1, -1),
                type_embedding.reshape(1, -1)
            )[0][0]
            similarities[q_type] = round(similarity, 3)
            
        return similarities

    def batch_identify_question_types(self, questions):
        """批量处理多个问题"""
        if not questions:
            return []
        
        # 一次性编码所有问题 → 只有1个进度条（或关闭进度条）
        question_embeddings = self.model.encode(questions, show_progress_bar=False)
        
        results = []
        for i, question in enumerate(questions):
            question_embedding = question_embeddings[i]
            result = self._classify_with_embedding(question_embedding, question)
            results.append(result)
        
        return results
    
    def _classify_with_embedding(self, question_embedding, question):
        """使用预计算的嵌入向量进行分类"""
        similarities = {}
        for q_type, type_embedding in self.type_embeddings.items():
            similarity = cosine_similarity(
                question_embedding.reshape(1, -1),
                type_embedding.reshape(1, -1)
            )[0][0]
            similarities[q_type] = similarity
        
        # 特殊处理否定/例外问题
        if self._is_exception_question(question):
            similarities['exception'] = max(similarities.get('exception', 0), 0.8)
        
        # 选择相似度超过阈值的类型
        matched_types = []
        for q_type, similarity in similarities.items():
            if similarity >= self.similarity_threshold:
                matched_types.append((q_type, similarity))
        
        matched_types.sort(key=lambda x: x[1], reverse=True)
        
        if matched_types:
            result_types = [matched_types[0][0]]
            if (len(matched_types) > 1 and 
                matched_types[1][1] >= matched_types[0][1] * 0.85):
                result_types.append(matched_types[1][0])
            return result_types
        else:
            return ['general']

# ========================= 层次化图谱构建_预计算向量管理器 =========================
class PrecomputedVectorManager:
    def __init__(self, entity_embeddings, keyword_embeddings):
        """利用现有的预计算向量"""
        self.entity_embeddings = entity_embeddings
        self.keyword_embeddings = keyword_embeddings
        
        # 建立快速查找索引
        self.entity_to_idx = {entity: idx for idx, entity in enumerate(entity_embeddings['entities'])}
        self.keyword_to_idx = {keyword: idx for idx, keyword in enumerate(keyword_embeddings['keywords'])}
        
        # 预计算医学概念中心向量
        self._compute_medical_concept_centers()
    
    def _compute_medical_concept_centers(self):
        """计算各医学概念类型的中心向量"""
        concept_keywords = {
            'disease': ['disease', 'syndrome', 'disorder', 'condition', 'is_a', 'subtype'],
            'symptom': ['symptom', 'sign', 'manifestation', 'presents', 'shows'],
            'treatment': ['treat', 'therapy', 'medication', 'drug', 'cure']
        }
        
        self.concept_centers = {}
        
        for concept, keywords in concept_keywords.items():
            vectors = []
            for keyword in keywords:
                if keyword in self.keyword_to_idx:
                    idx = self.keyword_to_idx[keyword]
                    vectors.append(self.keyword_embeddings['embeddings'][idx])
            
            if vectors:
                # 计算中心向量（平均值）
                center_vector = np.mean(vectors, axis=0)
                self.concept_centers[concept] = center_vector
                
        logger.info(f"Computed concept centers for {len(self.concept_centers)} medical concepts")
    
    def get_entity_vector(self, entity):
        """获取实体向量"""
        if entity in self.entity_to_idx:
            idx = self.entity_to_idx[entity]
            return self.entity_embeddings['embeddings'][idx]
        return None
    
    def get_keyword_vector(self, keyword):
        """获取关键词向量"""
        if keyword in self.keyword_to_idx:
            idx = self.keyword_to_idx[keyword]
            return self.keyword_embeddings['embeddings'][idx]
        return None
    
    def batch_entity_similarity(self, entities, concept_type):
        """批量计算实体与概念中心的相似度"""
        if concept_type not in self.concept_centers:
            return {}
        
        concept_center = self.concept_centers[concept_type]
        similarities = {}
        
        entity_vectors = []
        valid_entities = []
        
        for entity in entities:
            vector = self.get_entity_vector(entity)
            if vector is not None:
                entity_vectors.append(vector)
                valid_entities.append(entity)
        
        if entity_vectors:
            # 批量计算相似度（矩阵运算）
            entity_matrix = np.array(entity_vectors)
            concept_center = concept_center.reshape(1, -1)
            
            sims = cosine_similarity(entity_matrix, concept_center).flatten()
            
            for entity, sim in zip(valid_entities, sims):
                similarities[entity] = sim
        
        return similarities

# ========================= 层次化图谱构建_重写核心语义匹配逻辑 =========================

class OptimizedSemanticMatcher:
    def __init__(self, vector_manager):
        self.vector_manager = vector_manager
        # 使用统一配置的阈值
        self.thresholds = {
            'disease': THRESHOLDS.get_threshold('medical_concept', 'disease'),
            'symptom': THRESHOLDS.get_threshold('medical_concept', 'symptom'),
            'treatment': THRESHOLDS.get_threshold('medical_concept', 'treatment')
        }
    
    def classify_triple_batch(self, triples, concept_type):
        """批量分类三元组是否属于指定概念类型"""
        matched_triples = []
        
        # 提取所有实体
        all_entities = set()
        for triple in triples:
            if len(triple) >= 3:
                all_entities.add(triple[0])  # head entity
                all_entities.add(triple[2])  # tail entity
        
        # 批量计算实体相似度
        entity_similarities = self.vector_manager.batch_entity_similarity(
            list(all_entities), concept_type
        )
        
        # 批量计算关系相似度
        relations = [triple[1] for triple in triples if len(triple) >= 3]
        relation_similarities = self._batch_relation_similarity(relations, concept_type)
        
        threshold = self.thresholds[concept_type]
        
        for triple in triples:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                # 语义匹配为主导
                head_sim = entity_similarities.get(head, 0)
                tail_sim = entity_similarities.get(tail, 0)
                relation_sim = relation_similarities.get(relation, 0)
                
                # 综合相似度评分
                combined_score = max(head_sim, tail_sim) * 0.6 + relation_sim * 0.4
                
                if combined_score >= threshold:
                    matched_triples.append({
                        'triple': triple,
                        'score': combined_score,
                        'head_sim': head_sim,
                        'tail_sim': tail_sim,
                        'relation_sim': relation_sim
                    })
        
        return matched_triples
    
    def _batch_relation_similarity(self, relations, concept_type):
        """批量计算关系相似度"""
        if concept_type not in self.vector_manager.concept_centers:
            return {}
        
        concept_center = self.vector_manager.concept_centers[concept_type]
        similarities = {}
        
        relation_vectors = []
        valid_relations = []
        
        for relation in relations:
            vector = self.vector_manager.get_keyword_vector(relation)
            if vector is not None:
                relation_vectors.append(vector)
                valid_relations.append(relation)
        
        if relation_vectors:
            relation_matrix = np.array(relation_vectors)
            concept_center = concept_center.reshape(1, -1)
            
            sims = cosine_similarity(relation_matrix, concept_center).flatten()
            
            for relation, sim in zip(valid_relations, sims):
                similarities[relation] = sim
        
        return similarities


# ========================= 优化的层次化图谱构建类 =========================

class OptimizedHierarchicalKGFramework:
    def __init__(self, entity_embeddings, keyword_embeddings, use_semantic_matching=True):
        # 保持原有数据结构
        self.disease_hierarchy = defaultdict(list)
        self.symptom_hierarchy = defaultdict(list)
        self.treatment_hierarchy = defaultdict(list)
        
        self.hierarchy_weights = {
            'is_a': 1.0,
            'part_of': 0.9,
            'subtype_of': 0.95,
            'category_of': 0.8,
            'related_to': 0.6
        }
        
        # 新的优化组件
        self.vector_manager = PrecomputedVectorManager(entity_embeddings, keyword_embeddings)
        self.use_semantic_matching = use_semantic_matching

        # 使用统一配置的阈值
        self.thresholds = {
            'disease': THRESHOLDS.get_threshold('medical_concept', 'disease'),
            'symptom': THRESHOLDS.get_threshold('medical_concept', 'symptom'),
            'treatment': THRESHOLDS.get_threshold('medical_concept', 'treatment')
        }
    
    def build_hierarchical_structure(self, flat_kg):
        """优化的层次结构构建 - 一次遍历同时构建所有层次"""
        if not ABLATION_CONFIG['USE_HIERARCHICAL_KG']:
            logger.info("Hierarchical KG Framework disabled in ablation study")
            return
            
        logger.info("Building optimized hierarchical knowledge structure with single-pass approach...")
        
        # 一次性构建所有层次
        self._build_all_hierarchies_single_pass(flat_kg)
        
        logger.info(f"Built optimized hierarchies: diseases={len(self.disease_hierarchy)}, "
                   f"symptoms={len(self.symptom_hierarchy)}, treatments={len(self.treatment_hierarchy)}")
    
    def _build_all_hierarchies_single_pass(self, flat_kg):
        """一次遍历，同时构建所有概念类型的层次"""
        
        # 预提取所有唯一实体和关系，避免重复提取
        all_entities = set()
        all_relations = set()
        for triple in flat_kg:
            if len(triple) >= 3:
                all_entities.update([triple[0], triple[2]])
                all_relations.add(triple[1])
        
        logger.info(f"Extracted {len(all_entities)} unique entities and {len(all_relations)} unique relations")
        
        # 一次性计算所有实体与各概念中心的相似度
        entity_similarities = {}
        for concept_type in ['disease', 'symptom', 'treatment']:
            entity_similarities[concept_type] = self.vector_manager.batch_entity_similarity(
                list(all_entities), concept_type
            )
        
        # 一次性计算所有关系与各概念中心的相似度
        relation_similarities = {}
        for concept_type in ['disease', 'symptom', 'treatment']:
            relation_similarities[concept_type] = self._batch_relation_similarity(
                list(all_relations), concept_type
            )
        
        logger.info("Completed batch similarity computation for all concept types")
        
        # 一次遍历，同时评估和构建所有层次
        for triple in tqdm(flat_kg, desc="Building all hierarchies simultaneously"):
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                # 同时计算与所有概念类型的相似度
                concept_scores = {}
                for concept_type in ['disease', 'symptom', 'treatment']:
                    head_sim = entity_similarities[concept_type].get(head, 0)
                    tail_sim = entity_similarities[concept_type].get(tail, 0)
                    relation_sim = relation_similarities[concept_type].get(relation, 0)
                    
                    # 综合相似度评分
                    combined_score = max(head_sim, tail_sim) * 0.6 + relation_sim * 0.4
                    concept_scores[concept_type] = combined_score
                
                # 根据阈值判断并直接添加到对应层次
                for concept_type, score in concept_scores.items():
                    if score >= self.thresholds[concept_type]:
                        self._add_to_hierarchy(triple, concept_type, score)
        
        # 对所有层次按语义得分排序
        self._sort_all_hierarchies()
    
    def _batch_relation_similarity(self, relations, concept_type):
        """批量计算关系相似度"""
        if concept_type not in self.vector_manager.concept_centers:
            return {}
        
        concept_center = self.vector_manager.concept_centers[concept_type]
        similarities = {}
        
        relation_vectors = []
        valid_relations = []
        
        for relation in relations:
            vector = self.vector_manager.get_keyword_vector(relation)
            if vector is not None:
                relation_vectors.append(vector)
                valid_relations.append(relation)
        
        if relation_vectors:
            relation_matrix = np.array(relation_vectors)
            concept_center = concept_center.reshape(1, -1)
            
            sims = cosine_similarity(relation_matrix, concept_center).flatten()
            
            for relation, sim in zip(valid_relations, sims):
                similarities[relation] = sim
        
        return similarities
    
    def _add_to_hierarchy(self, triple, concept_type, score):
        """将三元组添加到指定概念类型的层次中"""
        head, relation, tail = triple[0], triple[1], triple[2]
        
        hierarchy_dict = getattr(self, f"{concept_type}_hierarchy")
        
        # 保持与原代码一致的权重设置
        if concept_type == 'disease':
            default_weight = 0.5
        else:
            default_weight = 0.7  # symptom和treatment使用0.7
        
        hierarchy_item = {
            'entity': None,
            'relation': relation,
            'weight': self.hierarchy_weights.get(relation.lower(), default_weight),
            'semantic_score': score
        }
        
        # 根据概念类型确定层次结构
        if concept_type == 'disease':
            # 疾病层次：子类 -> 父类
            hierarchy_item['entity'] = head
            hierarchy_dict[tail].append(hierarchy_item)
        else:
            # 症状/治疗层次：实体 -> 相关项
            hierarchy_item['entity'] = tail
            hierarchy_dict[head].append(hierarchy_item)
    
    def _sort_all_hierarchies(self):
        """对所有层次按语义得分排序"""
        for hierarchy_name in ['disease_hierarchy', 'symptom_hierarchy', 'treatment_hierarchy']:
            hierarchy_dict = getattr(self, hierarchy_name)
            for entity, items in hierarchy_dict.items():
                items.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)
    
    def get_hierarchical_context(self, entity, context_type='all'):
        """保持原有接口不变"""
        if not ABLATION_CONFIG['USE_HIERARCHICAL_KG']:
            return {}
            
        context = {}
        
        if context_type in ['all', 'disease']:
            context['diseases'] = self.disease_hierarchy.get(entity, [])
        
        if context_type in ['all', 'symptom']:
            context['symptoms'] = self.symptom_hierarchy.get(entity, [])
        
        if context_type in ['all', 'treatment']:
            context['treatments'] = self.treatment_hierarchy.get(entity, [])
        
        return context
    
    def print_hierarchy_stats(self):
        """打印层次图谱的详细统计信息"""
        
        total_disease_relations = sum(len(items) for items in self.disease_hierarchy.values())
        total_symptom_relations = sum(len(items) for items in self.symptom_hierarchy.values())
        total_treatment_relations = sum(len(items) for items in self.treatment_hierarchy.values())
        
        logger.info(f"📊 Detailed Hierarchy Statistics:")
        logger.info(f"  Diseases: {len(self.disease_hierarchy)} parent nodes, {total_disease_relations} total relations")
        logger.info(f"  Symptoms: {len(self.symptom_hierarchy)} parent nodes, {total_symptom_relations} total relations") 
        logger.info(f"  Treatments: {len(self.treatment_hierarchy)} parent nodes, {total_treatment_relations} total relations")
        
        # 显示一些示例
        if self.disease_hierarchy:
            sample_disease = list(self.disease_hierarchy.keys())[0]
            logger.info(f"  Example - {sample_disease}: {len(self.disease_hierarchy[sample_disease])} sub-items")
        
        total_relations = total_disease_relations + total_symptom_relations + total_treatment_relations
        logger.info(f"  Total hierarchical relations: {total_relations}")


        
# ========================= 多策略实体链接 =========================

class SemanticMatcher:
    # 语义匹配器类：通过计算实体间的语义相似度进行匹配
    
    def __init__(self):
        # 构造函数，初始化相似度阈值
        self.similarity_threshold = THRESHOLDS.get_threshold('semantic_matching', 'jaccard_similarity')
        # 从全局阈值配置中获取Jaccard相似度阈值（默认0.7）
        # 这个阈值用于判断两个实体是否足够相似以建立链接
        # 先用余弦相似度、候选Jaccard相似度
    
    def match(self, entities, umls_kg):
        # 主要匹配方法
        # entities: 待匹配的实体列表（如：["alzheimer", "dementia"]）
        # umls_kg: UMLS知识图谱中的标准概念列表
        
        """语义相似度匹配"""
        
        if not ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
            # 检查消融实验配置，如果多策略链接被禁用则直接返回
            return {}
            
        matches = {}
        # 初始化匹配结果字典
        
        for entity in entities:
            # 遍历每个待匹配的实体
            best_match = None
            # 存储当前实体的最佳匹配结果
            best_score = 0
            # 存储当前实体的最高匹配分数
            
            for kg_entity in umls_kg:
                # 遍历知识图谱中的每个标准概念
                score = self._calculate_hybrid_similarity(entity, kg_entity)
                # 计算当前实体与知识图谱概念的语义相似度
                
                if score > best_score and score > self.similarity_threshold:
                    # 如果当前分数既超过历史最高分又超过设定阈值
                    best_score = score
                    # 更新最高分数
                    best_match = kg_entity
                    # 更新最佳匹配概念
            
            if best_match:
                # 如果找到了满足条件的匹配
                matches[entity] = {'match': best_match, 'score': best_score, 'method': 'semantic'}
                # 将匹配结果存储到字典中，包含匹配的概念、分数和方法标识
        
        return matches
        # 返回所有实体的语义匹配结果
    
    def _calculate_hybrid_similarity(self, entity1, entity2):
        """混合相似度计算"""
        # 尝试使用预训练向量
        vector_sim = self._calculate_vector_cosine_similarity(entity1, entity2)
        if vector_sim is not None:
            return vector_sim
        
        # 回退到改进的文本余弦相似度
        text_cosine = self._calculate_cosine_similarity(entity1, entity2)
        jaccard = self._calculate_jaccard_similarity(entity1, entity2)
        
        # 组合两种文本相似度
        return 0.7 * text_cosine + 0.3 * jaccard

class ContextAwareLinker:
    # 上下文感知链接器：基于实体在问题上下文中的相关性进行链接
    
    def __init__(self):
        # 构造函数
        self.context_weight = 0.3
        # 设置上下文权重为0.3（虽然在当前实现中未直接使用）
    
    def link(self, entities, context):
        # 上下文感知链接方法
        # entities: 待链接的实体列表
        # context: 完整的问题文本作为上下文
        
        """上下文感知链接"""
        
        if not ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
            # 检查多策略链接是否启用
            return {}
            
        links = {}
        # 初始化链接结果字典
        
        context_words = set(context.lower().split())
        # 将问题文本转为小写分词，创建上下文词汇集合
        # 例如："What causes Alzheimer disease?" -> {"what", "causes", "alzheimer", "disease"}
        
        for entity in entities:
            # 遍历每个待链接的实体
            entity_words = set(entity.lower().split())
            # 将实体转为小写分词，创建实体词汇集合
            
            context_overlap = len(entity_words.intersection(context_words))
            # 计算实体词汇与上下文词汇的重叠数量
            
            context_score = context_overlap / len(entity_words) if entity_words else 0
            # 计算上下文分数：重叠词汇数 / 实体总词汇数
            # 这表示实体有多少比例的词汇出现在问题上下文中
            
            links[entity] = {
                'context_score': context_score,
                'method': 'context_aware'
            }
            # 存储每个实体的上下文链接结果
        
        return links
        # 返回所有实体的上下文感知链接结果
        
class ConfidenceEstimator:
    # 置信度估计器：融合多种匹配策略的结果并计算最终置信度
    
    def __init__(self):
        # 构造函数，设置融合权重
        self.weight_semantic = 0.6
        # 语义匹配的权重为0.6（占主导地位）
        self.weight_context = 0.4
        # 上下文匹配的权重为0.4（起辅助作用）
    
    def fuse_results(self, semantic_matches, context_matches):
        # 结果融合方法
        # semantic_matches: 语义匹配器的输出结果
        # context_matches: 上下文感知链接器的输出结果
        
        """置信度估计和融合"""
        
        if not ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
            # 检查多策略链接是否启用
            return {}
            
        final_links = {}
        # 初始化最终链接结果字典
        
        all_entities = set(semantic_matches.keys()) | set(context_matches.keys())
        # 使用集合并运算获取所有出现过的实体
        # 确保即使某个实体只在一种匹配中出现也会被处理
        
        for entity in all_entities:
            # 遍历所有实体
            semantic_score = semantic_matches.get(entity, {}).get('score', 0)
            # 安全地获取语义匹配分数，如果不存在则默认为0
            # 使用链式get()调用避免KeyError异常
            
            context_score = context_matches.get(entity, {}).get('context_score', 0)
            # 安全地获取上下文匹配分数，如果不存在则默认为0
            
            combined_score = (self.weight_semantic * semantic_score + 
                            self.weight_context * context_score)
            # 计算加权组合分数：0.6 * 语义分数 + 0.4 * 上下文分数
            # 这样语义匹配的影响更大，上下文匹配起到调节作用
            
            final_links[entity] = {
                'final_score': combined_score,
                'semantic_score': semantic_score,
                'context_score': context_score,
                'method': 'fused'
            }
            # 存储融合后的完整结果，包含最终分数、各项分数和方法标识
        
        return final_links
        # 返回所有实体的融合链接结果

class EnhancedEntityLinking:
    # 定义增强的实体链接类，用于将医学文本中的实体链接到标准化的UMLS概念
    
    def __init__(self):
        # 类的构造函数，初始化三个核心组件
        self.semantic_matcher = SemanticMatcher()
        # 初始化语义匹配器：通过计算实体间的语义相似度进行匹配
        # 使用Jaccard相似度等方法比较实体的语义特征
        
        self.context_aware_linker = ContextAwareLinker()
        # 初始化上下文感知链接器：考虑实体在问题上下文中的相关性
        # 分析实体与问题上下文词汇的重叠度，提供上下文匹配分数
        
        self.confidence_estimator = ConfidenceEstimator()
        # 初始化置信度估计器：融合多种匹配策略的结果并计算最终置信度
        # 将语义匹配和上下文匹配的结果进行加权融合
    
    def multi_strategy_linking(self, entities, context, umls_kg):
        # 定义多策略实体链接方法
        # 参数说明：
        # - entities: 待链接的实体列表（从问题中提取的医学术语）
        # - context: 上下文信息（完整的问题文本）
        # - umls_kg: UMLS知识图谱（标准化的医学概念集合）
        
        """多策略实体链接"""
        # 方法的中文描述注释
        
        if not ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
            # 检查消融实验配置：如果多策略链接功能被禁用
            # ABLATION_CONFIG是用于控制不同模块开关的实验配置
            return {}
            # 返回空字典，跳过多策略链接处理
            
        semantic_matches = self.semantic_matcher.match(entities, umls_kg)
        # 调用语义匹配器进行第一轮匹配
        # 计算每个实体与UMLS知识图谱中概念的语义相似度
        # 返回格式：{entity: {'match': matched_concept, 'score': similarity_score, 'method': 'semantic'}}
        
        context_matches = self.context_aware_linker.link(entities, context)
        # 调用上下文感知链接器进行第二轮匹配
        # 基于实体在问题上下文中的相关性进行链接
        # 返回格式：{entity: {'context_score': overlap_score, 'method': 'context_aware'}}
        
        final_links = self.confidence_estimator.fuse_results(
            semantic_matches, context_matches
        )
        # 调用置信度估计器融合两种匹配结果
        # 将语义匹配分数和上下文匹配分数按权重组合
        # 计算最终的置信度分数：final_score = 0.6 * semantic_score + 0.4 * context_score
        
        return final_links
        # 返回融合后的最终链接结果
        # 格式：{entity: {'final_score': combined_score, 'semantic_score': score1, 'context_score': score2, 'method': 'fused'}}

# ========================= 自适应UMLS知识选择 =========================

class AdaptiveUMLSSelector:
    def __init__(self, umls_api):
        self.umls_api = umls_api
        self.task_specific_weights = {
            'treatment': {
                'therapeutic_procedure': 2.0,
                'pharmacologic_substance': 1.8,
                'clinical_drug': 1.6
            },
            'diagnosis': {
                'disease_syndrome': 2.0,
                'sign_symptom': 1.8,
                'finding': 1.6
            },
            'causation': {
                'disease_syndrome': 1.8,
                'pathologic_function': 1.6,
                'injury_poisoning': 1.4
            },
            'prevention': {
                'therapeutic_procedure': 1.8,
                'preventive_procedure': 2.0,
                'health_care_activity': 1.6
            }
        }
    
    def select_relevant_umls_knowledge(self, question_type, entities):
        """根据问题类型选择相关UMLS知识"""
        if not ABLATION_CONFIG['USE_ADAPTIVE_UMLS']:
            return self.get_general_knowledge(entities)
            
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
        treatment_knowledge = []
        
        for entity in entities:
            concepts = self.umls_api.search_concepts(entity)
            if concepts and 'results' in concepts:
                for concept in concepts['results'][:5]:
                    cui = concept.get('ui', '')
                    relations = self.umls_api.get_concept_relations(cui)
                    
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
        diagnosis_knowledge = []
        
        for entity in entities:
            concepts = self.umls_api.search_concepts(entity)
            if concepts and 'results' in concepts:
                for concept in concepts['results'][:5]:
                    cui = concept.get('ui', '')
                    relations = self.umls_api.get_concept_relations(cui)
                    
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
        causation_knowledge = []
        
        for entity in entities:
            concepts = self.umls_api.search_concepts(entity)
            if concepts and 'results' in concepts:
                for concept in concepts['results'][:5]:
                    cui = concept.get('ui', '')
                    relations = self.umls_api.get_concept_relations(cui)
                    
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
        prevention_knowledge = []
        
        for entity in entities:
            concepts = self.umls_api.search_concepts(entity)
            if concepts and 'results' in concepts:
                for concept in concepts['results'][:5]:
                    cui = concept.get('ui', '')
                    relations = self.umls_api.get_concept_relations(cui)
                    
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
        general_knowledge = []
        
        for entity in entities:
            # 1. 对每个实体搜索UMLS概念
            concepts = self.umls_api.search_concepts(entity)
            if concepts and 'results' in concepts:
                # 2. 只取前3个最相关的概念
                for concept in concepts['results'][:3]:
                    cui = concept.get('ui', '')
                    # 3. 获取该概念的所有关系
                    relations = self.umls_api.get_concept_relations(cui)
                    # 4. 每个概念只取前10个关系
                    general_knowledge.extend(relations[:10])
        
        return general_knowledge

# ========================= 知识图谱引导的思维链推理 =========================

class SchemaReasoner:
    def __init__(self):
        self.medical_schemas = {
            'diagnosis': ['symptom', 'finding', 'test', 'disease'],
            'treatment': ['disease', 'medication', 'procedure', 'outcome'],
            'causation': ['risk_factor', 'cause', 'disease', 'complication'],
            'prevention': ['risk_factor', 'intervention', 'prevention', 'outcome']
        }
    
    def infer_paths(self, question, kg):
        """基于模式推理路径"""
        if not ABLATION_CONFIG['USE_KG_GUIDED_REASONING']:
            return []
            
        question_type = self._identify_question_schema(question)
        schema = self.medical_schemas.get(question_type, [])
        
        reasoning_paths = []
        for i in range(len(schema) - 1):
            start_type = schema[i]
            end_type = schema[i + 1]
            paths = self._find_schema_paths(kg, start_type, end_type)
            reasoning_paths.extend(paths)
        
        return reasoning_paths
    
    def _identify_question_schema(self, question):
        """识别问题模式"""
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ['treat', 'therapy', 'medication']):
            return 'treatment'
        elif any(keyword in question_lower for keyword in ['cause', 'why', 'due to']):
            return 'causation'
        elif any(keyword in question_lower for keyword in ['prevent', 'avoid', 'reduce risk']):
            return 'prevention'
        else:
            return 'diagnosis'
    
    def _find_schema_paths(self, kg, start_type, end_type):
        """查找符合模式的路径"""
        paths = []
        for triple in kg:
            if len(triple) >= 3:
                if (start_type in triple[0].lower() or start_type in triple[1].lower()) and \
                   (end_type in triple[2].lower() or end_type in triple[1].lower()):
                    paths.append(triple)
        return paths

class KGGuidedReasoningEngine:
    def __init__(self, kg, llm):
        self.kg = kg
        self.llm = llm
        self.schema_reasoner = SchemaReasoner()
    
    def kg_guided_reasoning(self, question, kg_subgraph):
        """知识图谱引导的推理"""
        if not ABLATION_CONFIG['USE_KG_GUIDED_REASONING']:
            return "KG-guided reasoning disabled in ablation study"
            
        schema_paths = self.schema_reasoner.infer_paths(question, self.kg)
        optimal_subgraph = self.generate_optimal_subgraph(
            question, schema_paths, kg_subgraph
        )
        reasoning_result = self.llm_reasoning_with_kg(question, optimal_subgraph)
        
        return reasoning_result
    
    def generate_optimal_subgraph(self, question, schema_paths, kg_subgraph):
        """生成最优子图"""
        combined_graph = kg_subgraph + schema_paths
        
        scored_triples = []
        for triple in combined_graph:
            score = self._calculate_relevance_score(question, triple)
            scored_triples.append((triple, score))
        
        scored_triples.sort(key=lambda x: x[1], reverse=True)
        optimal_subgraph = [triple for triple, score in scored_triples[:15]]
        
        return optimal_subgraph
    
    def _calculate_relevance_score(self, question, triple):
        """计算三元组与问题的相关性分数"""
        question_words = set(question.lower().split())
        triple_words = set()
        
        for element in triple:
            triple_words.update(element.lower().split())
        
        overlap = len(question_words.intersection(triple_words))
        relevance_score = overlap / len(question_words) if question_words else 0
        
        return relevance_score
    
    def llm_reasoning_with_kg(self, question, kg_subgraph):
        """使用LLM进行知识图谱增强推理"""
        kg_context = "\n".join([f"{t[0]} -> {t[1]} -> {t[2]}" for t in kg_subgraph])
        
        prompt = f"""
        Question: {question}
        
        Knowledge Graph Context:
        {kg_context}
        
        Based on the structured medical knowledge above, provide step-by-step reasoning to answer the question.
        Focus on the relationships and pathways shown in the knowledge graph.
        """
        
        try:
            response = self.llm([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Error in LLM reasoning: {e}")
            return "Unable to generate reasoning based on knowledge graph."

# ========================= 优化多跳推理 =========================

class PathRanker:
    def __init__(self):
        self.medical_relation_weights = {
            'causes': 3.0,
            'treats': 2.8,
            'prevents': 2.5,
            'symptom_of': 2.2,
            'diagnoses': 2.0,
            'associated_with': 1.8,
            'located_in': 1.5,
            'part_of': 1.2,
            'related_to': 1.0
        }
    
    def rank_by_quality(self, paths):
        """根据质量对路径进行排序"""
        if not ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP']:
            return paths
            
        scored_paths = []
        
        for path in paths:
            quality_score = self._calculate_path_quality(path)
            scored_paths.append((path, quality_score))
        
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        return [path for path, score in scored_paths]
    
    def _calculate_path_quality(self, path):
        """计算路径质量分数"""
        if not path:
            return 0
        
        relation_score = 0
        for step in path:
            if len(step) >= 2:
                relation = step[1].lower()
                for key, weight in self.medical_relation_weights.items():
                    if key in relation:
                        relation_score += weight
                        break
                else:
                    relation_score += 0.5
        
        length_penalty = len(path) * 0.1
        quality_score = relation_score - length_penalty
        
        return quality_score

class OptimizedMultiHopReasoning:
    def __init__(self, kg, path_ranker=None):
        self.kg = kg
        self.path_ranker = path_ranker or PathRanker()
        self.reasoning_cache = {}
    
    def intelligent_path_selection(self, start_entities, target_entities, max_hops=3):
        """智能路径选择"""
        if not ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP']:
            return self._basic_path_selection(start_entities, target_entities, max_hops)
            
        weighted_paths = self.calculate_medical_relevance_weights(
            start_entities, target_entities
        )
        
        pruned_paths = self.dynamic_pruning(weighted_paths, max_hops)
        quality_ranked_paths = self.path_ranker.rank_by_quality(pruned_paths)
        
        return quality_ranked_paths
    
    def _basic_path_selection(self, start_entities, target_entities, max_hops):
        """基础版本的路径选择"""
        basic_paths = []
        for start_entity in start_entities:
            for target_entity in target_entities:
                paths = self._find_connecting_paths(start_entity, target_entity)
                basic_paths.extend(paths[:3])
        return basic_paths
    
    def calculate_medical_relevance_weights(self, start_entities, target_entities):
        """计算基于医学知识的路径权重"""
        weighted_paths = []
        
        for start_entity in start_entities:
            for target_entity in target_entities:
                cache_key = f"{start_entity}-{target_entity}"
                
                if cache_key in self.reasoning_cache:
                    weighted_paths.extend(self.reasoning_cache[cache_key])
                    continue
                
                paths = self._find_connecting_paths(start_entity, target_entity)
                
                for path in paths:
                    weight = self._calculate_medical_relevance(path)
                    weighted_paths.append((path, weight))
                
                self.reasoning_cache[cache_key] = [(path, weight) for path, weight in weighted_paths[-len(paths):]]
        
        return weighted_paths
    
    def dynamic_pruning(self, weighted_paths, max_hops):
        """动态剪枝策略"""
        pruned_paths = []
        
        weighted_paths.sort(key=lambda x: x[1], reverse=True)
        
        for path, weight in weighted_paths:
            if len(path) <= max_hops:
                if weight > 0.5:
                    pruned_paths.append(path)
            
            if len(pruned_paths) >= 20:
                break
        
        return pruned_paths
    
    def _find_connecting_paths(self, start_entity, target_entity):
        """查找连接路径"""
        paths = []
        
        for triple in self.kg:
            if len(triple) >= 3:
                if triple[0] == start_entity and triple[2] == target_entity:
                    paths.append([triple])
        
        intermediate_entities = set()
        for triple in self.kg:
            if len(triple) >= 3 and triple[0] == start_entity:
                intermediate_entities.add(triple[2])
        
        for intermediate in intermediate_entities:
            for triple in self.kg:
                if len(triple) >= 3 and triple[0] == intermediate and triple[2] == target_entity:
                    first_hop = next((t for t in self.kg if len(t) >= 3 and t[0] == start_entity and t[2] == intermediate), None)
                    if first_hop:
                        paths.append([first_hop, triple])
        
        return paths[:10]
    
    def _calculate_medical_relevance(self, path):
        """计算医学相关性"""
        relevance_score = 0
        
        for step in path:
            if len(step) >= 3:
                entity_score = self._get_entity_medical_score(step[0]) + self._get_entity_medical_score(step[2])
                relation_score = self._get_relation_medical_score(step[1])
                relevance_score += entity_score + relation_score
        
        return relevance_score / len(path) if path else 0
    
    def _get_entity_medical_score(self, entity):
        """获取实体的医学相关性分数"""
        medical_keywords = ['disease', 'symptom', 'treatment', 'medication', 'diagnosis', 'therapy']
        entity_lower = entity.lower()
        
        score = 0
        for keyword in medical_keywords:
            if keyword in entity_lower:
                score += 1
        
        return score
    
    def _get_relation_medical_score(self, relation):
        """获取关系的医学相关性分数"""
        relation_weights = {
            'causes': 3.0, 'treats': 2.8, 'prevents': 2.5,
            'symptom_of': 2.2, 'diagnoses': 2.0, 'associated_with': 1.8
        }
        
        relation_lower = relation.lower()
        for key, weight in relation_weights.items():
            if key in relation_lower:
                return weight
        
        return 1.0

# ========================= UMLS API集成 =========================

class UMLS_API:
    def __init__(self, api_key, version="current"):
        """初始化UMLS API客户端"""
        self.api_key = api_key
        self.version = version
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        
        self.session = requests.Session()
        self.session.timeout = 30
        
        self.cache = {}
        self.cache_size = 10000
        self.failed_cuis = set()
        
        if ABLATION_CONFIG['USE_UMLS_NORMALIZATION'] or ABLATION_CONFIG['USE_ADAPTIVE_UMLS']:
            try:
                self._test_connection()
                logger.info("UMLS API connection successful")
            except Exception as e:
                logger.warning(f"UMLS API connection failed: {e}. Operating in offline mode.")
        else:
            logger.info("🔬 UMLS API disabled in ablation study")
    
    def _test_connection(self):
        """测试API连接"""
        try:
            params = {
                "string": "pain",
                "apiKey": self.api_key,
                "pageNumber": 1,
                "pageSize": 1
            }
            response = self.session.get(self.search_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            if 'result' not in data:
                raise Exception("Invalid API response format")
        except Exception as e:
            raise Exception(f"API connection test failed: {e}")
    
    def search_concepts(self, search_string, search_type="words", page_size=25):
        """搜索UMLS概念"""
        if not (ABLATION_CONFIG['USE_UMLS_NORMALIZATION'] or ABLATION_CONFIG['USE_ADAPTIVE_UMLS']):
            return None
            
        cache_key = f"search_{search_string}_{page_size}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            params = {
                "string": search_string,
                "apiKey": self.api_key,
                "pageNumber": 1,
                "pageSize": page_size
            }
            
            response = self.session.get(self.search_url, params=params)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            data = response.json()
            result = data.get("result", {})
            
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching concepts for '{search_string}': {e}")
            return None
    
    def get_concept_details(self, cui):
        """获取概念详细信息"""
        if not (ABLATION_CONFIG['USE_UMLS_NORMALIZATION'] or ABLATION_CONFIG['USE_ADAPTIVE_UMLS']):
            return None
            
        cache_key = f"details_{cui}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            url = f"{self.content_url}/CUI/{cui}"
            params = {"apiKey": self.api_key}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            response.encoding = "utf-8"
            
            data = response.json()
            result = data.get("result", {})
            
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting details for CUI {cui}: {e}")
            return None
    
    def get_concept_atoms(self, cui):
        """获取概念的原子信息"""
        if not (ABLATION_CONFIG['USE_UMLS_NORMALIZATION'] or ABLATION_CONFIG['USE_ADAPTIVE_UMLS']):
            return None
            
        cache_key = f"atoms_{cui}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            url = f"{self.content_url}/CUI/{cui}/atoms"
            params = {"apiKey": self.api_key, "pageSize": 100}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            response.encoding = "utf-8"
            
            data = response.json()
            result = data.get("result", [])
            
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting atoms for CUI {cui}: {e}")
            return None
    
    def get_concept_relations(self, cui):
        """获取概念关系"""
        if not (ABLATION_CONFIG['USE_UMLS_NORMALIZATION'] or ABLATION_CONFIG['USE_ADAPTIVE_UMLS']):
            return []
            
        cache_key = f"relations_{cui}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if cui in self.failed_cuis:
            return []
        
        all_relations = []
        
        try:
            for page in range(1, 6):
                url = f"{self.content_url}/CUI/{cui}/relations"
                params = {
                    "apiKey": self.api_key,
                    "pageNumber": page,
                    "pageSize": 100
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                response.encoding = "utf-8"
                
                data = response.json()
                page_relations = data.get("result", [])
                
                if not page_relations:
                    break
                    
                all_relations.extend(page_relations)
            
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = all_relations
            
            return all_relations
            
        except requests.exceptions.HTTPError as e:
            if "404" in str(e):
                self.failed_cuis.add(cui)
                logger.warning(f"CUI {cui} not found (404), adding to failed cache")
            else:
                logger.error(f"HTTP error getting relations for CUI {cui}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting relations for CUI {cui}: {e}")
            return []

class UMLSNormalizer:
    def __init__(self, api_key):
        """初始化UMLS标准化器"""
        # 创建UMLS API实例，用于与UMLS数据库交互
        self.umls_api = UMLS_API(api_key)
        
        # 本地缓存字典，存储已查询过的术语结果，避免重复API调用
        self.local_cache = {}
        
        # 语义类型缓存，存储概念的语义类型信息
        self.semantic_type_cache = {}
        
        # 引用全局的层次化知识图谱框架实例
        # 用于获取概念的层次结构信息
        self.hierarchical_kg = hierarchical_kg_framework
        
        # 创建增强实体链接器实例，用于多策略实体链接
        self.enhanced_entity_linking = EnhancedEntityLinking()
        
        # 创建自适应UMLS选择器实例，根据问题类型选择相关知识
        self.adaptive_umls_selector = AdaptiveUMLSSelector(self.umls_api)
        
        # 定义语义类型的优先级权重字典
        # UMLS中不同语义类型的重要性排序，数值越高优先级越高
        self.semantic_type_priority = {
            'T047': 10,  # Disease or Syndrome - 疾病或综合征，最高优先级
            'T184': 9,   # Sign or Symptom - 体征或症状
            'T061': 8,   # Therapeutic or Preventive Procedure - 治疗或预防程序
            'T121': 7,   # Pharmacologic Substance - 药理物质
            'T023': 6,   # Body Part, Organ, or Organ Component - 身体部位、器官或组件
            'T037': 5,   # Injury or Poisoning - 伤害或中毒
            'T046': 4,   # Pathologic Function - 病理功能
            'T033': 3,   # Finding - 发现
            'T170': 2,   # Intellectual Product - 智力产品
            'T169': 1    # Functional Concept - 功能概念，最低优先级
        }
    
    def _get_best_cui_for_term(self, term):
        """为给定术语获取最佳CUI"""
        # 检查是否启用了UMLS标准化功能（消融实验控制）
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return None
            
        # 首先检查本地缓存，避免重复API调用
        if term in self.local_cache:
            return self.local_cache[term]
        
        try:
            # 调用UMLS API搜索与term相关的概念
            search_results = self.umls_api.search_concepts(term)
            
            # 检查API返回结果是否有效
            if not search_results or 'results' not in search_results:
                return None
            
            # 获取搜索结果列表
            results = search_results['results']
            if not results:
                return None
            
            # 初始化最佳匹配变量
            best_cui = None      # 最佳匹配的CUI
            best_score = 0       # 最高匹配分数
            
            # 遍历所有搜索结果，找到最佳匹配
            for result in results:
                cui = result['ui']        # 获取概念的唯一标识符(CUI)
                name = result['name']     # 获取概念的名称
                
                # 计算当前结果与原术语的匹配分数
                score = self._calculate_match_score(term, name, result)
                
                # 如果当前分数更高，更新最佳匹配
                if score > best_score:
                    best_score = score
                    best_cui = cui
            
            # 将结果缓存到本地，避免重复查询
            self.local_cache[term] = best_cui
            return best_cui
            
        except Exception as e:
            # 异常处理：记录错误日志并返回None
            logger.error(f"Error getting CUI for term '{term}': {e}")
            return None
    
    def _calculate_match_score(self, original_term, concept_name, result):
        """计算匹配分数"""
        score = 0  # 初始化分数
        
        # 完全匹配检查（不区分大小写）
        if original_term.lower() == concept_name.lower():
            score += 100  # 完全匹配给予最高分数
        # 原术语包含在概念名称中
        elif original_term.lower() in concept_name.lower():
            score += 50   # 部分包含给予中等分数
        # 概念名称包含在原术语中
        elif concept_name.lower() in original_term.lower():
            score += 30   # 反向包含给予较低分数
        
        # 词汇重叠度计算
        # 将术语和概念名称分别拆分为单词集合
        original_words = set(original_term.lower().split())
        concept_words = set(concept_name.lower().split())
        # 计算交集（共同单词数量）
        overlap = len(original_words & concept_words)
        # 每个重叠单词贡献10分
        score += overlap * 10
        
        # 词根匹配检查
        if self._has_root_match(original_term, concept_name):
            score += 20  # 词根匹配额外加分
        
        return score
    
    def _has_root_match(self, term1, term2):
        """检查词根匹配"""
        # 定义常见英语后缀列表
        suffixes = ['s', 'es', 'ing', 'ed', 'er', 'est', 'ly']
        
        def get_root(word):
            """提取单词词根的内部函数"""
            # 遍历后缀列表
            for suffix in suffixes:
                # 如果单词以某个后缀结尾
                if word.endswith(suffix):
                    # 去除后缀返回词根
                    return word[:-len(suffix)]
            # 如果没有匹配的后缀，返回原单词
            return word
        
        # 获取两个术语的词根（转为小写）
        root1 = get_root(term1.lower())
        root2 = get_root(term2.lower())
        
        # 检查词根是否相等或互相包含
        return root1 == root2 or root1 in root2 or root2 in root1
    
    def get_concept_synonyms(self, cui):
        """获取概念的同义词"""
        # 检查UMLS标准化功能是否启用
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return []
            
        try:
            # 调用API获取概念的原子信息（atoms包含所有同义表达）
            atoms_result = self.umls_api.get_concept_atoms(cui)
            
            # 检查API返回结果
            if not atoms_result:
                return []
            
            # 初始化同义词列表
            synonyms = []
            # 遍历所有原子记录
            for atom in atoms_result:
                # 获取原子的名称
                name = atom.get('name', '')
                # 如果名称存在且不重复，添加到同义词列表
                if name and name not in synonyms:
                    synonyms.append(name)
            
            return synonyms
            
        except Exception as e:
            # 异常处理：记录错误并返回空列表
            logger.error(f"Error getting synonyms for CUI {cui}: {e}")
            return []
    
    def get_concept_relations(self, cui):
        """获取概念关系"""
        # 检查UMLS标准化功能是否启用
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return []
            
        try:
            # 调用API获取概念的所有关系
            relations_result = self.umls_api.get_concept_relations(cui)
            
            # 检查API返回结果
            if not relations_result:
                return []
            
            # 初始化关系列表
            relations = []
            # 遍历所有关系记录
            for relation in relations_result:
                # 提取关系的各个组件
                rel_type = relation.get('relationLabel', '')     # 关系类型标签
                related_cui = relation.get('relatedId', '')      # 相关概念的CUI
                related_name = relation.get('relatedIdName', '') # 相关概念的名称
                
                # 如果关系类型和相关CUI都存在
                if rel_type and related_cui:
                    # 构建关系字典并添加到列表
                    relations.append({
                        'relation_type': rel_type,
                        'related_cui': related_cui,
                        'related_name': related_name
                    })
            
            return relations
            
        except Exception as e:
            # 异常处理：记录错误并返回空列表
            logger.error(f"Error getting relations for CUI {cui}: {e}")
            return []
    
    def normalize_medical_terms(self, entities):
        """将医学术语标准化为UMLS概念"""
        # 检查UMLS标准化功能是否启用
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return entities  # 如果禁用，直接返回原实体列表
            
        # 初始化标准化后的实体列表
        normalized_entities = []
        
        # 遍历每个输入实体
        for entity in entities:
            try:
                # 为当前实体获取最佳匹配的CUI
                cui = self._get_best_cui_for_term(entity)
                
                # 如果找到了CUI
                if cui:
                    # 获取该CUI对应的概念详细信息
                    concept_details = self.umls_api.get_concept_details(cui)
                    
                    # 如果成功获取概念详情
                    if concept_details:
                        # 提取首选名称（UMLS标准名称）
                        preferred_name = concept_details.get('name', entity)
                        # 添加标准名称到结果列表
                        normalized_entities.append(preferred_name)
                        # 记录标准化过程的调试信息
                        logger.debug(f"标准化: {entity} -> {preferred_name} (CUI: {cui})")
                    else:
                        # 如果获取详情失败，保留原实体
                        normalized_entities.append(entity)
                else:
                    # 如果没找到CUI，保留原实体
                    normalized_entities.append(entity)
                    
            except Exception as e:
                # 异常处理：记录错误并保留原实体
                logger.error(f"Error normalizing entity '{entity}': {e}")
                normalized_entities.append(entity)
        
        return normalized_entities
    
    def get_semantic_variants(self, entity):
        """获取实体的语义变体"""
        # 检查UMLS标准化功能是否启用
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return [entity]  # 禁用时返回原实体
            
        try:
            # 获取实体对应的CUI
            cui = self._get_best_cui_for_term(entity)
            if not cui:
                return [entity]  # 没找到CUI，返回原实体
            
            # 获取同义词列表
            synonyms = self.get_concept_synonyms(cui)
            # 获取关系列表
            relations = self.get_concept_relations(cui)
            related_terms = []  # 相关术语列表
            
            # 从关系中提取特定类型的相关术语
            for relation in relations:
                # 只选择特定关系类型的相关术语
                if relation['relation_type'] in ['SY', 'PT', 'equivalent_to']:
                    related_terms.append(relation['related_name'])
            
            # 合并原实体、同义词和相关术语
            variants = [entity] + synonyms + related_terms
            
            # 去重处理
            unique_variants = []  # 唯一变体列表
            seen = set()          # 已见过的术语集合（小写）
            
            # 遍历所有变体进行去重
            for variant in variants:
                # 检查变体是否有效且未重复
                if variant and variant.lower() not in seen and len(variant) > 2:
                    seen.add(variant.lower())      # 记录小写版本
                    unique_variants.append(variant)  # 添加原始大小写版本
            
            # 最多返回10个变体，避免结果过多
            return unique_variants[:10]
            
        except Exception as e:
            # 异常处理：记录错误并返回原实体
            logger.error(f"Error getting semantic variants for '{entity}': {e}")
            return [entity]
    
    def get_concept_hierarchy(self, entity):
        """获取概念层次结构"""
        # 检查UMLS标准化功能是否启用
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return {}  # 禁用时返回空字典
            
        try:
            # 获取实体对应的CUI
            cui = self._get_best_cui_for_term(entity)
            if not cui:
                return {}  # 没找到CUI，返回空字典
            
            # 获取概念的所有关系
            relations = self.get_concept_relations(cui)
            
            # 初始化层次结构字典
            hierarchy = {
                'broader': [],   # 上位概念（更宽泛的概念）
                'narrower': [],  # 下位概念（更具体的概念）
                'related': []    # 相关概念（同级概念）
            }
            
            # 遍历所有关系，按类型分类
            for relation in relations:
                rel_type = relation['relation_type']      # 关系类型
                related_name = relation['related_name']   # 相关概念名称
                
                # 根据关系类型分类到相应类别
                if rel_type in ['RB', 'inverse_isa', 'parent']:
                    # 上位关系：当前概念是相关概念的子类
                    hierarchy['broader'].append(related_name)
                elif rel_type in ['RN', 'isa', 'child']:
                    # 下位关系：当前概念是相关概念的父类
                    hierarchy['narrower'].append(related_name)
                elif rel_type in ['RT', 'related_to']:
                    # 相关关系：概念之间存在关联但非层次关系
                    hierarchy['related'].append(related_name)
            
            return hierarchy
            
        except Exception as e:
            # 异常处理：记录错误并返回空字典
            logger.error(f"Error getting concept hierarchy for '{entity}': {e}")
            return {}
    
    def enhanced_entity_linking_method(self, entities, context, question_types):
        """增强的实体链接"""
        # 检查多策略链接功能是否启用
        if not ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
            return {}  # 禁用时返回空字典
            
        try:
            # 构建UMLS知识图谱
            umls_kg = []
            # 为每个实体搜索UMLS概念
            for entity in entities:
                concepts = self.umls_api.search_concepts(entity)
                # 如果搜索成功且有结果
                if concepts and 'results' in concepts:
                    # 提取前5个概念的名称加入知识图谱
                    umls_kg.extend([concept['name'] for concept in concepts['results'][:5]])
            
            # 调用增强实体链接器进行多策略链接
            linking_results = self.enhanced_entity_linking.multi_strategy_linking(
                entities,      # 待链接的实体列表
                context,       # 上下文信息
                umls_kg        # UMLS知识图谱
            )
            
            return linking_results
            
        except Exception as e:
            # 异常处理：记录错误并返回空字典
            logger.error(f"Error in enhanced entity linking: {e}")
            return {}
    
    def adaptive_knowledge_selection(self, question_types, entities):
        """自适应知识选择"""
        # 检查自适应UMLS功能是否启用
        if not ABLATION_CONFIG['USE_ADAPTIVE_UMLS']:
            return []  # 禁用时返回空列表
            
        try:
            # 初始化选中的知识列表
            selected_knowledge = []
            
            # 遍历所有问题类型
            for question_type in question_types:
                # 为每种问题类型选择相关的UMLS知识
                knowledge = self.adaptive_umls_selector.select_relevant_umls_knowledge(
                    question_type,  # 问题类型（如'treatment', 'diagnosis'）
                    entities        # 相关实体列表
                )
                # 将选中的知识扩展到总列表中
                selected_knowledge.extend(knowledge)
            
            return selected_knowledge
            
        except Exception as e:
            # 异常处理：记录错误并返回空列表
            logger.error(f"Error in adaptive knowledge selection: {e}")
            return []

# ========================= 医学推理规则模块 =========================

class MedicalReasoningRules:
    def __init__(self, umls_normalizer=None):
        """初始化医学推理规则"""
        self.umls_normalizer = umls_normalizer
        self.kg_guided_reasoning = None
        
        self.rules = {
            'transitivity': {
                'causes': ['causes', 'leads_to', 'results_in', 'induces'],
                'treats': ['treats', 'alleviates', 'improves', 'cures'],
                'part_of': ['part_of', 'located_in', 'component_of'],
                'precedes': ['precedes', 'before', 'prior_to'],
                'prevents': ['prevents', 'reduces_risk_of', 'protects_against']
            },
            'inverse_relations': {
                'causes': 'caused_by',
                'treats': 'treated_by',
                'part_of': 'contains',
                'precedes': 'follows',
                'prevents': 'prevented_by'
            },
            'semantic_implications': {
                'symptom_of': 'has_symptom',
                'risk_factor_for': 'has_risk_factor',
                'complication_of': 'has_complication'
            },
            'medical_hierarchies': {
                'disease_subtype': 'is_type_of',
                'anatomical_part': 'part_of_anatomy',
                'drug_class': 'belongs_to_class'
            }
        }
        
        self.confidence_weights = {
            'direct': 1.0,
            'transitive_1hop': 0.8,
            'transitive_2hop': 0.6,
            'inverse': 0.9,
            'semantic': 0.7,
            'hierarchical': 0.75
        }
    
    def initialize_kg_guided_reasoning(self, kg, llm):
        """初始化知识图谱引导推理"""
        if ABLATION_CONFIG['USE_KG_GUIDED_REASONING']:
            self.kg_guided_reasoning = KGGuidedReasoningEngine(kg, llm)
        else:
            logger.info("🔬 KG-guided reasoning disabled in ablation study")
    
    def apply_reasoning_rules(self, knowledge_triples, max_hops=2):
        """
        应用医学推理规则扩展知识
        
        这个方法通过应用多种逻辑推理规则来扩展原始的医学知识三元组，
        从有限的事实中推导出更多隐含的医学知识关系
        
        参数:
            knowledge_triples (list): 原始知识三元组列表
                                    格式: [['entity1', 'relation', 'entity2'], ...]
            max_hops (int): 传递推理的最大跳数，默认为2
        
        返回:
            list: 扩展后的去重知识三元组列表
        """
        
        # ==================== 第1步：消融实验配置检查 ====================
        if not ABLATION_CONFIG['USE_REASONING_RULES']:
            logger.info("🔬 Medical reasoning rules disabled in ablation study")
            return knowledge_triples
            # 如果推理规则功能被禁用（消融实验控制），直接返回原始三元组
            # 这允许研究人员比较有无推理规则的系统性能差异
            # 消融实验是AI研究中常用的方法，用于验证各模块的贡献
        
        # ==================== 第2步：初始化数据结构 ====================
        expanded_triples = knowledge_triples.copy()
        # 创建原始三元组的副本，避免修改原始数据
        # 后续所有推理得到的新三元组都会添加到这个列表中
        
        reasoning_log = []
        # 初始化推理日志，记录每种推理类型生成的三元组数量
        # 用于性能监控和调试分析
        
        # ==================== 第3步：传递性推理 ====================
        transitive_triples = self._apply_transitivity(knowledge_triples, max_hops)
        """
        传递性推理示例：
        原始事实:
        - ['阿尔茨海默病', 'causes', '记忆丧失']
        - ['记忆丧失', 'leads_to', '认知障碍']
        
        推理结果:
        - ['阿尔茨海默病', 'transitively_causes', '认知障碍']
        
        医学意义: 如果A导致B，B导致C，那么A间接导致C
        这在医学中很常见，疾病→症状→功能障碍的链条
        """
        
        expanded_triples.extend(transitive_triples)
        # 将传递性推理的结果添加到扩展三元组列表
        reasoning_log.extend([('transitivity', len(transitive_triples))])
        # 记录传递性推理生成的三元组数量
        
        # ==================== 第4步：逆关系推理 ====================
        inverse_triples = self._apply_inverse_relations(knowledge_triples)
        """
        逆关系推理示例：
        原始事实:
        - ['阿司匹林', 'treats', '头痛']
        
        推理结果:
        - ['头痛', 'treated_by', '阿司匹林']
        
        医学意义: 许多医学关系是双向的
        如果药物A治疗疾病B，那么疾病B被药物A治疗
        这增加了知识图谱的连通性和查询灵活性
        """
        
        expanded_triples.extend(inverse_triples)
        reasoning_log.extend([('inverse', len(inverse_triples))])
        
        # ==================== 第5步：语义蕴涵推理 ====================
        semantic_triples = self._apply_semantic_implications(knowledge_triples)
        """
        语义蕴涵推理示例：
        原始事实:
        - ['糖尿病', 'symptom_of', '高血糖']
        
        推理结果:
        - ['高血糖', 'has_symptom', '糖尿病']
        
        医学意义: 某些关系在语义上互相蕴涵
        如果X是Y的症状，那么Y具有症状X
        这基于医学术语的语义结构进行推理
        """
        
        expanded_triples.extend(semantic_triples)
        reasoning_log.extend([('semantic', len(semantic_triples))])
        
        # ==================== 第6步：层次化推理 ====================
        hierarchical_triples = self._apply_hierarchical_reasoning(knowledge_triples)
        """
        层次化推理示例：
        原始事实:
        - ['心肌梗死', 'is_type_of', '心脏病']
        - UMLS层次结构显示: 心肌梗死 → 冠心病 → 心脏病
        
        推理结果:
        - ['心肌梗死', 'is_subtype_of', '冠心病']
        - ['冠心病', 'is_subtype_of', '心脏病']
        
        医学意义: 利用UMLS等医学本体的层次结构
        推导疾病、解剖结构、药物等的上下位关系
        这丰富了概念间的分类学关系
        """
        
        expanded_triples.extend(hierarchical_triples)
        reasoning_log.extend([('hierarchical', len(hierarchical_triples))])
        
        # ==================== 第7步：去重处理 ====================
        unique_triples = self._deduplicate_triples(expanded_triples)
        """
        去重处理的必要性：
        
        问题: 不同推理规则可能生成相同的三元组
        例如:
        - 传递性推理: ['疾病A', 'causes', '症状B']
        - 逆关系推理: ['症状B', 'caused_by', '疾病A'] → ['疾病A', 'causes', '症状B']
        
        解决方案: 标准化并去重
        - 转换为小写进行比较
        - 使用集合数据结构快速去重
        - 保留原始格式的三元组
        """
        
        # ==================== 第8步：日志记录和性能监控 ====================
        logger.info(f"推理扩展: {reasoning_log}")
        logger.info(f"原始三元组: {len(knowledge_triples)}, 扩展后: {len(unique_triples)}")
        """
        日志输出示例:
        推理扩展: [('transitivity', 5), ('inverse', 12), ('semantic', 8), ('hierarchical', 15)]
        原始三元组: 25, 扩展后: 58
        
        信息解读:
        - 传递性推理生成了5个新三元组
        - 逆关系推理生成了12个新三元组
        - 语义蕴涵推理生成了8个新三元组
        - 层次化推理生成了15个新三元组
        - 总共从25个原始三元组扩展到58个（去重后）
        
        性能指标:
        - 扩展倍数: 58/25 = 2.32倍
        - 各推理类型的贡献比例可用于优化
        """
        
        return unique_triples
        # 返回扩展和去重后的知识三元组

    
    def _apply_transitivity(self, triples, max_hops):
        """应用传递性推理"""
        transitive_triples = []
        
        relation_graph = {}
        for triple in triples:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                if head not in relation_graph:
                    relation_graph[head] = []
                relation_graph[head].append((relation, tail))
        
        for rule_type, relation_variants in self.rules['transitivity'].items():
            transitive_triples.extend(
                self._find_transitive_paths(relation_graph, relation_variants, max_hops)
            )
        
        return transitive_triples
    
    def _find_transitive_paths(self, graph, relation_variants, max_hops):
        """查找传递性路径"""
        paths = []
        
        for start_entity in graph:
            for hop in range(1, max_hops + 1):
                paths.extend(
                    self._dfs_transitive_search(graph, start_entity, relation_variants, hop, [])
                )
        
        return paths
    
    def _dfs_transitive_search(self, graph, current_entity, target_relations, remaining_hops, path):
        """深度优先搜索传递性路径"""
        if remaining_hops == 0:
            return []
        
        results = []
        
        if current_entity in graph:
            for relation, next_entity in graph[current_entity]:
                if any(target_rel in relation.lower() for target_rel in target_relations):
                    new_path = path + [(current_entity, relation, next_entity)]
                    
                    if remaining_hops == 1:
                        if len(new_path) >= 2:
                            start = new_path[0][0]
                            end = new_path[-1][2]
                            inferred_relation = f"transitively_{target_relations[0]}"
                            results.append([start, inferred_relation, end])
                    else:
                        results.extend(
                            self._dfs_transitive_search(
                                graph, next_entity, target_relations, 
                                remaining_hops - 1, new_path
                            )
                        )
        
        return results
    
    def _apply_inverse_relations(self, triples):
        """应用逆关系推理"""
        inverse_triples = []
        
        for triple in triples:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                for forward_rel, inverse_rel in self.rules['inverse_relations'].items():
                    if forward_rel in relation.lower():
                        inverse_triples.append([tail, inverse_rel, head])
        
        return inverse_triples
    
    def _apply_semantic_implications(self, triples):
        """应用语义蕴涵推理"""
        semantic_triples = []
        
        for triple in triples:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                for source_rel, target_rel in self.rules['semantic_implications'].items():
                    if source_rel in relation.lower():
                        semantic_triples.append([tail, target_rel, head])
        
        return semantic_triples
    
    def _apply_hierarchical_reasoning(self, triples):
        """应用层次推理"""
        hierarchical_triples = []
        
        if not self.umls_normalizer:
            return hierarchical_triples
        
        entities = set()
        for triple in triples:
            if len(triple) >= 3:
                entities.add(triple[0])
                entities.add(triple[2])
        
        for entity in entities:
            try:
                hierarchy = self.umls_normalizer.get_concept_hierarchy(entity)
                
                for broader_concept in hierarchy.get('broader', []):
                    hierarchical_triples.append([entity, 'is_subtype_of', broader_concept])
                
                for narrower_concept in hierarchy.get('narrower', []):
                    hierarchical_triples.append([narrower_concept, 'is_subtype_of', entity])
                
            except Exception as e:
                logger.error(f"Error in hierarchical reasoning for {entity}: {e}")
        
        return hierarchical_triples
    
    def _deduplicate_triples(self, triples):
        """去重三元组"""
        seen = set()
        unique_triples = []
        
        for triple in triples:
            if len(triple) >= 3:
                triple_key = (triple[0].lower(), triple[1].lower(), triple[2].lower())
                if triple_key not in seen:
                    seen.add(triple_key)
                    unique_triples.append(triple)
        
        return unique_triples

# ========================= 多跳推理模块 =========================

class MultiHopReasoning:
    def __init__(self, max_hops=3, umls_normalizer=None):
        """初始化多跳推理器"""
        self.max_hops = max_hops
        self.umls_normalizer = umls_normalizer
        self.reasoning_chains = []
        self.evidence_weights = {
            'direct': 1.0,
            'one_hop': 0.8,
            'two_hop': 0.6,
            'three_hop': 0.4
        }
        
        self.optimized_multi_hop = OptimizedMultiHopReasoning(kg=[], path_ranker=PathRanker())
    
    def perform_multi_hop_reasoning(self, question, kg_subgraph):
        """执行多跳推理"""
        if not ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP']:
            return self._basic_multi_hop_reasoning(question, kg_subgraph)
            
        self.optimized_multi_hop.kg = kg_subgraph
        
        question_entities = self._extract_question_entities(question)
        
        if self.umls_normalizer and ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            normalized_entities = self.umls_normalizer.normalize_medical_terms(question_entities)
            question_entities.extend(normalized_entities)
        
        if len(question_entities) >= 2:
            start_entities = question_entities[:1]
            target_entities = question_entities[1:]
            
            intelligent_paths = self.optimized_multi_hop.intelligent_path_selection(
                start_entities, target_entities, self.max_hops
            )
            
            reasoning_chains = []
            for path in intelligent_paths[:5]:
                chain = self._build_reasoning_chain_from_path(path, kg_subgraph)
                if chain:
                    reasoning_chains.append(chain)
        else:
            reasoning_chains = []
            for entity in question_entities:
                chain = self._build_reasoning_chain(entity, kg_subgraph, self.max_hops)
                if chain:
                    reasoning_chains.append(chain)
        
        final_answer = self._fuse_reasoning_chains(reasoning_chains, question)
        return final_answer
    
    def _basic_multi_hop_reasoning(self, question, kg_subgraph):
        """基础版本的多跳推理"""
        logger.info("🔬 Using basic multi-hop reasoning (optimized version disabled)")
        
        question_entities = self._extract_question_entities(question)
        
        if len(question_entities) >= 2:
            reasoning_summary = f"Basic reasoning: Found entities {question_entities[:2]} in knowledge graph."
            
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
        """从路径构建推理链"""
        chain = {
            'path': path,
            'confidence': self._calculate_path_confidence(path),
            'reasoning_steps': []
        }
        
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
        """计算路径置信度"""
        if not path:
            return 0.0
        
        total_confidence = 1.0
        for step in path:
            if len(step) >= 2:
                relation_weight = self._calculate_relation_weight(step[1])
                total_confidence *= relation_weight
        
        length_penalty = 0.9 ** len(path)
        return total_confidence * length_penalty
    
    def _extract_question_entities(self, question):
        """从问题中提取实体"""
        entities = []
        
        medical_terms = [
            'alzheimer', 'dementia', 'brain', 'memory', 'cognitive',
            'treatment', 'medication', 'symptom', 'diagnosis', 'disease',
            'protein', 'amyloid', 'tau', 'hippocampus', 'cortex'
        ]
        
        question_lower = question.lower()
        for term in medical_terms:
            if term in question_lower:
                entities.append(term)
        
        words = question.split()
        for word in words:
            if word[0].isupper() and len(word) > 3:
                entities.append(word)
        
        return list(set(entities))
    
    def _build_reasoning_chain(self, start_entity, kg_subgraph, max_hops):
        """构建从起始实体开始的推理链"""
        chain = {
            'start_entity': start_entity,
            'paths': [],
            'confidence': 0.0
        }
        
        graph = self._build_graph_from_subgraph(kg_subgraph)
        
        for hop in range(1, max_hops + 1):
            hop_paths = self._find_paths_at_hop(graph, start_entity, hop)
            chain['paths'].extend(hop_paths)
        
        chain['confidence'] = self._calculate_chain_confidence(chain['paths'])
        
        return chain
    
    def _build_graph_from_subgraph(self, kg_subgraph):
        """从子图构建图结构"""
        graph = {}
        
        for triple in kg_subgraph:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                if head not in graph:
                    graph[head] = []
                graph[head].append({
                    'relation': relation,
                    'target': tail,
                    'weight': self._calculate_relation_weight(relation)
                })
        
        return graph
    
    def _find_paths_at_hop(self, graph, start_entity, target_hop):
        """查找指定跳数的路径"""
        def dfs_path_search(current_entity, current_hop, path, visited):
            if current_hop == target_hop:
                return [path]
            
            if current_entity not in graph or current_entity in visited:
                return []
            
            visited.add(current_entity)
            paths = []
            
            for edge in graph[current_entity]:
                new_path = path + [(current_entity, edge['relation'], edge['target'])]
                paths.extend(
                    dfs_path_search(edge['target'], current_hop + 1, new_path, visited.copy())
                )
            
            return paths
        
        return dfs_path_search(start_entity, 0, [], set())
    
    def _calculate_relation_weight(self, relation):
        """计算关系权重"""
        relation_lower = relation.lower().replace('_', ' ')
        
        weights = {
            'causes': 3.0, 'treats': 2.8, 'prevents': 2.5,
            'associated_with': 2.2, 'diagnoses': 2.0,
            'symptom_of': 1.8, 'risk_factor': 1.6,
            'interacts_with': 1.4, 'located_in': 1.2,
            'part_of': 1.0, 'related_to': 0.8
        }
        
        for key, weight in weights.items():
            if key in relation_lower:
                return weight
        
        return 1.0
    
    def _calculate_chain_confidence(self, paths):
        """计算推理链的置信度"""
        if not paths:
            return 0.0
        
        total_confidence = 0.0
        for path in paths:
            path_confidence = 1.0
            hop_weight = self.evidence_weights.get(f"{len(path)}_hop", 0.2)
            
            for step in path:
                relation_weight = self._calculate_relation_weight(step[1])
                path_confidence *= relation_weight
            
            path_confidence *= hop_weight
            total_confidence += path_confidence
        
        return min(total_confidence / len(paths), 1.0)
    
    def _fuse_reasoning_chains(self, reasoning_chains, question):
        """融合推理结果"""
        if not reasoning_chains:
            return "Unable to find sufficient reasoning paths."
        
        reasoning_chains.sort(key=lambda x: x['confidence'], reverse=True)
        
        answer_components = []
        total_confidence = 0.0
        
        for chain in reasoning_chains[:3]:
            if chain['confidence'] > 0.1:
                chain_summary = self._summarize_chain(chain)
                answer_components.append(chain_summary)
                total_confidence += chain['confidence']
        
        if answer_components:
            final_answer = f"Based on multi-hop reasoning (confidence: {total_confidence:.2f}):\n"
            final_answer += "\n".join(answer_components)
            return final_answer
        else:
            return "Insufficient evidence for multi-hop reasoning."
    
    def _summarize_chain(self, chain):
        """总结推理链"""
        summary = f"From {chain['start_entity']}:"
        
        best_paths = sorted(chain['paths'], 
                           key=lambda p: self._calculate_path_score(p), 
                           reverse=True)[:2]
        
        for i, path in enumerate(best_paths):
            path_str = " -> ".join([f"{step[0]} ({step[1]}) {step[2]}" for step in path])
            summary += f"\nPath {i+1}: {path_str}"
        
        return summary
    
    def _calculate_path_score(self, path):
        """计算路径得分"""
        score = 1.0
        for step in path:
            score *= self._calculate_relation_weight(step[1])
        return score / len(path)

# ========================= 医学领域知识库 =========================

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

QUESTION_TYPE_KEYWORDS = {
    'definition': ['what is', 'define', 'definition', 'meaning'],
    'causation': ['cause', 'causes', 'reason', 'why', 'due to', 'because'],
    'treatment': ['treat', 'treatment', 'therapy', 'cure', 'medication', 'drug'],
    'symptom': ['symptom', 'sign', 'present', 'manifestation'],
    'diagnosis': ['diagnose', 'diagnosis', 'test', 'examination'],
    'prevention': ['prevent', 'prevention', 'avoid', 'reduce risk'],
    'exception': ['except', 'not', 'exclude', 'excluding', 'other than']
}

NEGATION_WORDS = ['not', 'except', 'excluding', 'other than', 'rather than', 'instead of', 'exclude']

dataset2processor = {
    'medmcqa': medmcqaZeroshotsProcessor,
    'medqa':medqaZeroshotsProcessor,
    'mmlu': mmluZeroshotsProcessor,
    'qa4mre':qa4mreZeroshotsProcessor
}
datasets = ['medqa', 'medmcqa', 'mmlu', 'qa4mre']



# ========================= 性能优化函数 =========================

def cleanup_resources(sample_count):
    """性能优化：定期清理系统资源"""
    try:
        collected = gc.collect()
        
        if hasattr(umls_normalizer, 'umls_api') and hasattr(umls_normalizer.umls_api, 'cache'):
            cache_size_before = len(umls_normalizer.umls_api.cache)
            if cache_size_before > MAX_CACHE_SIZE:
                cache_items = list(umls_normalizer.umls_api.cache.items())
                umls_normalizer.umls_api.cache = dict(cache_items[-KEEP_CACHE_SIZE:])
                logger.info(f"🧹 Cleaned UMLS cache: {cache_size_before} → {len(umls_normalizer.umls_api.cache)}")
        
        if hasattr(umls_normalizer, 'local_cache'):
            local_cache_size_before = len(umls_normalizer.local_cache)
            if local_cache_size_before > MAX_CACHE_SIZE:
                cache_items = list(umls_normalizer.local_cache.items())
                umls_normalizer.local_cache = dict(cache_items[-KEEP_CACHE_SIZE:])
                logger.info(f"🧹 Cleaned local cache: {local_cache_size_before} → {len(umls_normalizer.local_cache)}")
        
        if hasattr(umls_normalizer, 'umls_api') and hasattr(umls_normalizer.umls_api, 'failed_cuis'):
            failed_cuis_size_before = len(umls_normalizer.umls_api.failed_cuis)
            if failed_cuis_size_before > MAX_FAILED_CUIS:
                umls_normalizer.umls_api.failed_cuis.clear()
                logger.info(f"🧹 Cleaned failed CUI cache: {failed_cuis_size_before} → 0")
        
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
    """扩展医学缩写词"""
    expanded_text = text
    for abbr, full_form in MEDICAL_ABBREVIATIONS.items():
        pattern = r'\b' + re.escape(abbr) + r'\b'
        expanded_text = re.sub(pattern, full_form, expanded_text, flags=re.IGNORECASE)
    return expanded_text

def get_medical_synonyms(entity):
    """
    获取医学术语的同义词
    
    参数:
        entity (str): 输入的医学术语
    
    返回:
        list: 包含原术语及其所有同义词的列表
    """
    
    # 第1步：将输入的医学术语转换为小写，便于后续的字符串匹配
    # 这样可以避免大小写敏感的问题
    entity_lower = entity.lower()
    
    # 第2步：初始化同义词列表，将原始术语作为第一个元素
    # 确保即使没找到其他同义词，也会返回原术语
    synonyms = [entity]
    
    # 第3步：第一次UMLS（统一医学语言系统）规范化处理
    # 检查配置中是否启用了UMLS规范化功能
    if ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
        try:
            # 调用UMLS规范化器获取语义变体
            # 语义变体是指意思相同但表达方式不同的术语
            umls_variants = umls_normalizer.get_semantic_variants(entity)
            
            # 将获取到的UMLS变体添加到同义词列表中
            synonyms.extend(umls_variants)
            
            # 记录调试信息，显示找到的UMLS变体
            logger.debug(f"UMLS variants for '{entity}': {umls_variants}")
            
        except Exception as e:
            # 如果UMLS处理出现异常，记录错误信息但不中断程序执行
            logger.error(f"Error getting UMLS variants for '{entity}': {e}")
    
    # 第4步：从预定义的医学同义词词典中查找匹配项
    # 遍历词典中的每个键值对
    for key, synonym_list in MEDICAL_SYNONYMS.items():
        # 检查两种匹配情况：
        # 1. 词典的键包含在输入术语中（部分匹配）
        # 2. 输入术语包含在词典的同义词列表中（完全匹配）
        if key in entity_lower or entity_lower in synonym_list:
            # 如果找到匹配，将对应的同义词列表添加到结果中
            synonyms.extend(synonym_list)
    
    # 第5步：第二次UMLS规范化处理
    # 对已收集的所有同义词进行进一步的规范化
    if ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
        try:
            # 对当前收集到的所有同义词进行标准化处理
            # 这可能会生成更多的标准化形式
            normalized_synonyms = umls_normalizer.normalize_medical_terms(synonyms)
            
            # 将规范化后的术语添加到同义词列表中
            synonyms.extend(normalized_synonyms)
            
        except Exception as e:
            # 如果规范化处理出现异常，记录错误信息
            logger.error(f"Error normalizing synonyms for '{entity}': {e}")
    
    # 第6步：返回最终结果
    # 使用set()去除重复项，然后转换回list
    # 这确保每个同义词只出现一次
    return list(set(synonyms))

def has_negation(question):
    """检查问题是否包含否定词"""
    question_lower = question.lower()
    return any(neg_word in question_lower for neg_word in NEGATION_WORDS)

def calculate_relation_weight(relation_type):
    """计算关系重要性权重"""
    relation_lower = relation_type.lower().replace('_', ' ')
    
    if relation_lower in RELATION_IMPORTANCE_WEIGHTS:
        return RELATION_IMPORTANCE_WEIGHTS[relation_lower]
    
    for key, weight in RELATION_IMPORTANCE_WEIGHTS.items():
        if key in relation_lower or relation_lower in key:
            return weight
    
    return 1.0

def calculate_knowledge_quality_score(knowledge_items):
    """计算知识质量分数"""
    if not knowledge_items:
        return 0.0
    
    quality_scores = []
    
    for item in knowledge_items:
        score = 1.0
        
        if isinstance(item, list) and len(item) >= 3:
            entity, relation, objects = item[0], item[1], item[2]
            
            if len(entity) > 3:
                score += 0.5
            
            relation_weight = calculate_relation_weight(relation)
            score += relation_weight * 0.3
            
            object_count = len(objects.split(',')) if ',' in objects else 1
            score += min(object_count * 0.1, 1.0)
        
        quality_scores.append(score)
    
    return np.mean(quality_scores)

def convert_numpy_types(obj):
    """递归转换NumPy类型为Python原生类型"""
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
    """添加重试机制的装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for retry in range(max_retries):
                try:
                    result = func(*args, **kwargs)
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

def validate_knowledge_triple(head, relation, tail):
    """验证知识三元组的质量"""
    if pd.isna(head) or pd.isna(relation) or pd.isna(tail):
        return False
    
    head = str(head).strip() if head is not None else ""
    relation = str(relation).strip() if relation is not None else ""
    tail = str(tail).strip() if tail is not None else ""
    
    if not head or not relation or not tail:
        return False
    
    if len(head) < 2 or len(tail) < 2:
        return False
    
    noise_patterns = ['http', 'www', '@', '#', '___', '...', 'nan', 'none']
    for pattern in noise_patterns:
        if pattern in head.lower() or pattern in tail.lower():
            return False
    
    return True

def basic_entity_matching(question_kg, entity_embeddings, keyword_embeddings, question_text=""):
    """基础版本的实体匹配，用于消融实验"""
    match_kg = []
    entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
    entity_confidence_scores = []
    
    # 使用统一阈值配置
    keyword_match_threshold = THRESHOLDS.get_threshold('semantic_matching', 'keyword_matching')
    similarity_threshold = THRESHOLDS.get_threshold('entity_matching', 'basic_similarity')
    
    for kg_entity in question_kg:
        try:
            if kg_entity in keyword_embeddings["keywords"]:
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
            else:
                best_match_idx = None
                best_similarity = 0
                for idx, keyword in enumerate(keyword_embeddings["keywords"]):
                    if kg_entity.lower() in keyword.lower():
                        # 原来: similarity = 0.8
                        # 现在: 使用配置的阈值
                        similarity = keyword_match_threshold  
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_idx = idx
                
                if best_match_idx is None:
                    continue
                keyword_index = best_match_idx
            
            kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

            kg_entity_emb_norm = kg_entity_emb / np.linalg.norm(kg_entity_emb)
            entity_embeddings_norm = entity_embeddings_emb.values / np.linalg.norm(entity_embeddings_emb.values, axis=1, keepdims=True)
            
            cos_similarities = np.dot(entity_embeddings_norm, kg_entity_emb_norm)
            
            best_idx = np.argmax(cos_similarities)
            similarity_score = cos_similarities[best_idx]
            
            # 原来: if similarity_score >= 0.6:
            # 现在: 使用配置的阈值
            if similarity_score >= similarity_threshold:
                candidate_entity = entity_embeddings["entities"][best_idx]
                if candidate_entity not in match_kg:
                    match_kg.append(candidate_entity)
                    entity_confidence_scores.append(float(similarity_score))
                    logger.debug(f"Basic matched: {kg_entity} -> {candidate_entity} (score: {similarity_score:.3f})")
                
        except Exception as e:
            logger.error(f"Error in basic entity matching for {kg_entity}: {e}")
            continue
    
    return match_kg, entity_confidence_scores

def debug_entity_matching_progress(step_info, question_kg, question_text, question_types=None, 
                                 expanded_entities=None, match_kg=None, confidence_scores=None,
                                 extra_info=None):
    """专门用于实体匹配的调试打印，带步骤提示"""
    print(f"\n{'='*80}")
    print(f"实体匹配的调试打印:🔍 {step_info}")
    print(f"{'='*80}")
    print(f"📝 问题文本: {question_text}")
    print(f"🔤 原始实体: {question_kg}")
    
    # 可选信息，根据传入参数决定是否显示
    if question_types:
        print(f"🏷️  问题类型: {question_types}")
    
    if expanded_entities:
        print(f"📈 扩展实体: {expanded_entities[:10]}..." if len(expanded_entities) > 10 else f"📈 扩展实体: {expanded_entities}")
    
    if match_kg is not None:
        print(f"✅ 匹配结果: {match_kg}")
        print(f"📊 匹配实体数量: {len(match_kg)}")
    
    if confidence_scores:
        print(f"📊 置信度: {[f'{score:.3f}' for score in confidence_scores]}")
        print(f"📊 平均置信度: {np.mean(confidence_scores):.3f}")
    
    if extra_info:
        print(f"ℹ️  额外信息: {extra_info}")
    
    print(f"{'='*80}")

def enhanced_entity_matching(question_kg, entity_embeddings, keyword_embeddings, question_text=""):
    """增强的实体匹配，集成真实UMLS API和新优化"""
    
    # ===== 第1步：检查消融实验配置，决定是否使用增强功能 =====
    if not any([
        ABLATION_CONFIG['USE_HIERARCHICAL_KG'],        # 是否使用层次化知识图谱
        ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING'], # 是否使用多策略链接
        ABLATION_CONFIG['USE_ADAPTIVE_UMLS'],          # 是否使用自适应UMLS
        ABLATION_CONFIG['USE_UMLS_NORMALIZATION'],     # 是否使用UMLS标准化
        ABLATION_CONFIG['USE_REASONING_RULES']         # 是否使用推理规则
    ]):
        # 如果所有增强功能都关闭，则使用基础版本
        logger.info("🔬 Using basic entity matching (all enhancements disabled)")
        return basic_entity_matching(question_kg, entity_embeddings, keyword_embeddings, question_text)
    
    # ===== 第2步：初始化变量 =====
    match_kg = []                                      # 存储匹配到的知识图谱实体
    entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])  # 转换为DataFrame便于计算
    entity_confidence_scores = []                      # 存储每个匹配的置信度分数
    
    # 打印信息
    debug_entity_matching_progress(
        "1.初始化", 
        question_kg, question_text, match_kg
    )

    # ===== 第3步：问题类型识别 =====
    question_types = semantic_question_classifier.identify_question_type(question_text)
    # 例如：question_text="What causes Alzheimer's?" → question_types=['causation']

    # 打印信息
    debug_entity_matching_progress(
        "2.问题类型识别", 
        question_kg, question_text, 
        question_types=question_types, 
        match_kg=match_kg
    )
    
    # ===== 第4步：实体扩展 - 医学缩写词处理 =====
    expanded_entities = []
    for kg_entity in question_kg:
        # 扩展医学缩写词（如 AD → Alzheimer Disease）
        expanded_entity = expand_medical_abbreviations(kg_entity)
        expanded_entities.append(expanded_entity)
        # 例如：kg_entity="AD" → expanded_entity="Alzheimer Disease"

        # 打印信息
        debug_entity_matching_progress(
            "3.实体扩展中的扩展医学缩写词", 
            question_kg, question_text, 
            question_types=question_types, 
            expanded_entities=expanded_entities, 
            match_kg=match_kg,
            extra_info=f"匹配的实体为：{kg_entity}，扩展的实体为{expanded_entity}"
        )
        
        # 如果启用UMLS标准化，获取同义词
        if ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            synonyms = get_medical_synonyms(kg_entity)
            expanded_entities.extend(synonyms)
            # 例如：kg_entity="alzheimer" → synonyms=["dementia", "alzheimer disease", "ad"]

            # 打印信息
            debug_entity_matching_progress(
                "4.UMLS标准化，获取同义词", 
                question_kg, question_text, 
                question_types=question_types, 
                expanded_entities=expanded_entities, 
                match_kg=match_kg,
                extra_info=f"匹配的实体为：{kg_entity}，获取的同义词为{synonyms}"
            )
    
    # ===== 第5步：多策略实体链接（如果启用） =====
    if ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
        try:
            # 使用语义匹配 + 上下文感知的增强链接
            enhanced_links = umls_normalizer.enhanced_entity_linking_method(
                expanded_entities, question_text, question_types
            )
            
            # 筛选高置信度的链接结果
            for entity, link_info in enhanced_links.items():
                if link_info.get('final_score', 0) > 0.6:  # 置信度阈值为0.6
                    expanded_entities.append(entity)

            # 打印信息
            debug_entity_matching_progress(
                "5.多策略实体链接", 
                question_kg, question_text, 
                question_types=question_types, 
                expanded_entities=expanded_entities, 
                match_kg=match_kg,
                extra_info=f"链接结果:{enhanced_links}"
            )
                    
        except Exception as e:
            logger.error(f"Error in enhanced entity linking: {e}")
    
    # ===== 第6步：自适应UMLS知识选择（如果启用） =====
    if ABLATION_CONFIG['USE_ADAPTIVE_UMLS']:
        try:
            # 根据问题类型选择相关的UMLS知识
            adaptive_knowledge = umls_normalizer.adaptive_knowledge_selection(
                question_types, expanded_entities
            )
            
            # 从自适应知识中提取相关实体名称
            for knowledge_item in adaptive_knowledge:
                if isinstance(knowledge_item, dict):
                    related_name = knowledge_item.get('related_name', '')
                    if related_name:
                        expanded_entities.append(related_name)
                        # 例如：从UMLS关系中提取到 "cognitive_impairment"

            # 打印信息
            debug_entity_matching_progress(
                "6.自适应UMLS知识选择", 
                question_kg, question_text, 
                question_types=question_types, 
                expanded_entities=expanded_entities, 
                match_kg=match_kg,
                extra_info=f"根据问题类型，选择到的UMLS知识:{adaptive_knowledge}"
            )
                            
        except Exception as e:
            logger.error(f"Error in adaptive knowledge selection: {e}")
    
    # ===== 第7步：基于推理规则的实体扩展（如果启用） =====
    if ABLATION_CONFIG['USE_REASONING_RULES']:
        try:
            # 创建临时三元组用于推理
            temp_triples = [[entity, 'mentions', 'question'] for entity in expanded_entities]
            # 应用医学推理规则（如传递性、逆关系等）
            reasoned_triples = medical_reasoning_rules.apply_reasoning_rules(temp_triples)
            
            # 从推理结果中提取新的实体
            for triple in reasoned_triples:
                if len(triple) >= 3:
                    expanded_entities.extend([triple[0], triple[2]])  # 添加头实体和尾实体

            # 打印信息
            debug_entity_matching_progress(
                "7.基于推理规则的实体扩展", 
                question_kg, question_text, 
                question_types=question_types, 
                expanded_entities=expanded_entities, 
                match_kg=match_kg
            )

        except Exception as e:
            logger.error(f"Error in reasoning-based entity expansion: {e}")

    # ===== 第8步：去重处理 =====
    seen = set()                    # 用于记录已见过的实体（小写）
    unique_entities = []            # 存储去重后的唯一实体
    for entity in expanded_entities:
        if entity.lower() not in seen:
            seen.add(entity.lower())
            unique_entities.append(entity)

    # 打印信息
    debug_entity_matching_progress(
        "8.去重处理", 
        question_kg, question_text, 
        question_types=question_types, 
        expanded_entities=expanded_entities, 
        match_kg=match_kg
    )
    
    # 打印扩展结果（用于调试）
    logger.info(f"Original entities: {question_kg}")
    logger.info(f"Expanded entities (with optimizations): {unique_entities[:10]}...")
    # 例如：Original: ["alzheimer"] → Expanded: ["Alzheimer Disease", "dementia", "cognitive_impairment", ...]
    
    # ===== 第9步：动态阈值调整 =====
    is_negation = has_negation(question_text)  # 检查是否有否定词
    if 'exception' in question_types or is_negation:
        # 对于否定/例外问题，使用更严格的阈值
        base_threshold = THRESHOLDS.get_threshold('entity_matching', 'min_similarity')
        similarity_threshold = THRESHOLDS.adjust_for_negation(base_threshold)
        # 例如：base_threshold=0.6 → similarity_threshold=0.6*0.8=0.48
    else:
        # 普通问题使用标准阈值
        similarity_threshold = THRESHOLDS.get_threshold('entity_matching', 'enhanced_similarity')
        # 例如：similarity_threshold=0.6

    # 打印信息
    debug_entity_matching_progress(
        "9.动态阈值调整", 
        question_kg, question_text, 
        question_types=question_types, 
        expanded_entities=expanded_entities, 
        match_kg=match_kg
    )
    
    # ===== 第10步：向量匹配过程 =====
    for kg_entity in unique_entities:
        try:
            # 尝试直接在关键词嵌入中找到实体
            if kg_entity in keyword_embeddings["keywords"]:
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
            else:
                # 如果直接匹配失败，进行模糊匹配
                best_match_idx = None
                best_similarity = 0
                for idx, keyword in enumerate(keyword_embeddings["keywords"]):
                    # 检查实体是否包含在关键词中，或关键词包含在实体中
                    if kg_entity.lower() in keyword.lower() or keyword.lower() in kg_entity.lower():
                        # 计算Jaccard相似度（交集/并集）
                        similarity = len(set(kg_entity.lower().split()) & set(keyword.lower().split())) / len(set(kg_entity.lower().split()) | set(keyword.lower().split()))
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_idx = idx
                
                # 如果没找到匹配或相似度太低，跳过这个实体
                if best_match_idx is None or best_similarity < 0.3:
                    continue
                keyword_index = best_match_idx
            
            # ===== 第11步：向量相似度计算 =====
            # 获取实体的嵌入向量
            kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

            # 向量标准化（归一化到单位长度）
            kg_entity_emb_norm = kg_entity_emb / np.linalg.norm(kg_entity_emb)
            entity_embeddings_norm = entity_embeddings_emb.values / np.linalg.norm(entity_embeddings_emb.values, axis=1, keepdims=True)
            
            # 计算余弦相似度
            cos_similarities = np.dot(entity_embeddings_norm, kg_entity_emb_norm)
            
            # ===== 第12步：Top-K候选选择 =====
            # 按相似度降序排列，取前5个候选
            top_indices = np.argsort(cos_similarities)[::-1]
            
            best_match_found = False
            # 遍历前5个最相似的候选实体
            for idx in top_indices[:5]:
                similarity_score = cos_similarities[idx]           # 相似度分数
                candidate_entity = entity_embeddings["entities"][idx]  # 候选实体名称
                
                # 检查是否满足阈值要求且未重复
                if (similarity_score >= similarity_threshold and 
                    candidate_entity not in match_kg):
                    match_kg.append(candidate_entity)              # 添加到匹配列表
                    entity_confidence_scores.append(float(similarity_score))  # 记录置信度
                    best_match_found = True
                    logger.debug(f"Matched: {kg_entity} -> {candidate_entity} (score: {similarity_score:.3f})")
                    break  # 找到一个高质量匹配就停止
            
            # 如果没找到高置信度匹配，记录警告
            if not best_match_found:
                logger.warning(f"No high-confidence match found for entity: {kg_entity}")
                
        except (ValueError, IndexError):
            # 实体不在关键词嵌入中
            logger.error(f"Entity {kg_entity} not found in keyword embeddings")
            continue
        except Exception as e:
            # 其他处理错误
            logger.error(f"Error processing entity {kg_entity}: {e}")
            continue

    # 打印信息
    debug_entity_matching_progress(
        "10.向量匹配", 
        question_kg, question_text, 
        question_types=question_types, 
        expanded_entities=expanded_entities, 
        match_kg=match_kg,
        extra_info=f"置信度：{entity_confidence_scores}"
    )
    
    # ===== 第13步：结果统计和返回 =====
    if entity_confidence_scores:
        avg_confidence = np.mean(entity_confidence_scores)
        logger.info(f"Entity matching average confidence: {avg_confidence:.3f}")
    
    return match_kg, entity_confidence_scores  # 返回匹配的实体列表和置信度分数

def enhanced_find_shortest_path(start_entity_name, end_entity_name, candidate_list, question_types=[]):
    """增强的路径查找，带有医学知识权重"""
    global exist_entity
    paths_with_scores = []
    
    with driver.session() as session:
        try:
            result = session.run(
                "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
                "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
                "RETURN p LIMIT 15",
                start_entity_name=start_entity_name,
                end_entity_name=end_entity_name
            )
            
            paths = []
            short_path = 0
            
            for record in result:
                path = record["p"]
                entities = []
                relations = []
                path_quality_score = 0
                
                for i in range(len(path.nodes)):
                    node = path.nodes[i]
                    entity_name = node["name"]
                    entities.append(entity_name)
                    
                    if i < len(path.relationships):
                        relationship = path.relationships[i]
                        relation_type = relationship.type
                        relations.append(relation_type)
                        
                        if any([ABLATION_CONFIG['USE_HIERARCHICAL_KG'], 
                               ABLATION_CONFIG['USE_REASONING_RULES'],
                               ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP']]):
                            relation_weight = calculate_relation_weight(relation_type)
                            path_quality_score += relation_weight
                            
                            if question_types:
                                if 'treatment' in question_types and 'treat' in relation_type.lower():
                                    path_quality_score += 1.0
                                elif 'causation' in question_types and 'cause' in relation_type.lower():
                                    path_quality_score += 1.0
                                elif 'symptom' in question_types and 'symptom' in relation_type.lower():
                                    path_quality_score += 1.0
                        else:
                            path_quality_score += 1.0
               
                path_str = ""
                for i in range(len(entities)):
                    entities[i] = entities[i].replace("_"," ")
                    
                    if entities[i] in candidate_list:
                        short_path = 1
                        exist_entity = entities[i]
                        path_quality_score += 3
                        
                    path_str += entities[i]
                    if i < len(relations):
                        relations[i] = relations[i].replace("_"," ")
                        path_str += "->" + relations[i] + "->"
                
                path_length = len(relations)
                length_penalty = path_length * 0.1 if ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP'] else 0
                final_score = path_quality_score - length_penalty
                
                paths_with_scores.append((path_str, final_score))
                
                if short_path == 1:
                    if ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP']:
                        paths_with_scores.sort(key=lambda x: x[1], reverse=True)
                    paths = [path[0] for path in paths_with_scores[:5]]
                    break
            
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
    """原始函数，使用增强实现"""
    return enhanced_find_shortest_path(start_entity_name, end_entity_name, candidate_list, question_types)

def combine_lists(*lists):
    combinations = list(itertools.product(*lists))
    results = []
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    return results

def enhanced_get_entity_neighbors(entity_name: str, disease_flag, question_types=[]) -> Tuple[List[List[str]], List[str]]:
    """增强的邻居提取，带有问题类型感知过滤"""
    disease = []
    neighbor_list = []
    
    if any([ABLATION_CONFIG['USE_ADAPTIVE_UMLS'], ABLATION_CONFIG['USE_REASONING_RULES']]):
        limit = 25 if any(q_type in ['treatment', 'causation'] for q_type in question_types) else 20
    else:
        limit = 10
    
    query = f"""
    MATCH (e:Entity)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type,
           collect(n.name) AS neighbor_entities
    ORDER BY size(collect(n.name)) DESC
    LIMIT {limit}
    """
    
    try:
        result = session.run(query, entity_name=entity_name)
        relation_quality_scores = {}
        
        for record in result:
            rel_type = record["relationship_type"]
            
            if disease_flag == 1 and rel_type == 'has_symptom':
                continue

            neighbors = record["neighbor_entities"]
            
            if ABLATION_CONFIG['USE_REASONING_RULES']:
                quality_score = calculate_relation_weight(rel_type)
                
                if question_types:
                    if 'treatment' in question_types and 'treat' in rel_type.lower():
                        quality_score += 1.0
                    elif 'causation' in question_types and 'cause' in rel_type.lower():
                        quality_score += 1.0
                    elif 'symptom' in question_types and 'symptom' in rel_type.lower():
                        quality_score += 1.0
            else:
                quality_score = 1.0
            
            if "disease" in rel_type.replace("_"," ").lower():
                disease.extend(neighbors)
                quality_score += 1.0
                
            filtered_neighbors = []
            for neighbor in neighbors:
                if validate_knowledge_triple(entity_name, rel_type, neighbor):
                    filtered_neighbors.append(neighbor)
            
            if filtered_neighbors:
                neighbor_entry = [entity_name.replace("_"," "), rel_type.replace("_"," "), 
                                ','.join([x.replace("_"," ") for x in filtered_neighbors])]
                neighbor_list.append(neighbor_entry)
                relation_quality_scores[len(neighbor_list)-1] = quality_score
        
        if relation_quality_scores and ABLATION_CONFIG['USE_REASONING_RULES']:
            sorted_indices = sorted(relation_quality_scores.keys(), 
                                  key=lambda k: relation_quality_scores[k], reverse=True)
            neighbor_list = [neighbor_list[i] for i in sorted_indices]
    
    except Exception as e:
        logger.error(f"Error getting entity neighbors: {e}")
    
    return neighbor_list, disease

def get_entity_neighbors(entity_name: str, disease_flag, question_types=[]) -> List[List[str]]:
    """原始函数签名保持不变"""
    neighbor_list, disease = enhanced_get_entity_neighbors(entity_name, disease_flag, question_types)
    return neighbor_list, disease

@retry_on_failure()
def prompt_path_finding(path_input):
    """原始路径查找提示模板"""
    template = """
    There are some knowledge graph path. They follow entity->relationship->entity format.
    \n\n
    {Path}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Path-based Evidence 1, Path-based Evidence 2,...\n\n

    Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["Path"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(Path = path_input)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(Path = path_input,\
                                                        text={})

    response = chat(chat_prompt_with_values.to_messages())
    if response.content is not None:
        return response.content
    else:
        return ""

@retry_on_failure()
def prompt_neighbor(neighbor):
    """原始邻居提示模板"""
    template = """
    There are some knowledge graph. They follow entity->relationship->entity list format.
    \n\n
    {neighbor}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2,...\n\n

    Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["neighbor"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(neighbor = neighbor)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(neighbor = neighbor,\
                                                        text={})

    response = chat(chat_prompt_with_values.to_messages())
    if response.content is not None:
        return response.content
    else:
        return ""

@retry_on_failure()
def self_knowledge_retrieval(graph, question):
    """原始知识检索提示模板"""
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

    prompt = PromptTemplate(
        template = template,
        input_variables = ["graph", "question"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(graph = graph, question=question)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(graph = graph, question=question,\
                                                        text={})

    response = chat(chat_prompt_with_values.to_messages())
    if response.content is not None:
        return response.content
    else:
        return ""

def enhanced_self_knowledge_retrieval_reranking(graph, question):
    """增强的知识重排序，带有医学知识感知和多跳推理"""
    
    if not any([ABLATION_CONFIG['USE_REASONING_RULES'], 
               ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP'],
               ABLATION_CONFIG['USE_KG_GUIDED_REASONING']]):
        logger.info("🔬 Using basic knowledge retrieval reranking")
        return self_knowledge_retrieval(graph, question)
    
    question_types = semantic_question_classifier.identify_question_type(question)
    has_neg = has_negation(question)
    
    if ABLATION_CONFIG['USE_OPTIMIZED_MULTIHOP']:
        try:
            graph_triples = []
            for line in graph.split('\n'):
                if '->' in line:
                    parts = line.split('->')
                    if len(parts) >= 3:
                        head = parts[0].strip()
                        relation = parts[1].strip()
                        tail = '->'.join(parts[2:]).strip()
                        graph_triples.append([head, relation, tail])
            
            if graph_triples:
                reasoned_result = multi_hop_reasoner.perform_multi_hop_reasoning(question, graph_triples)
                logger.debug(f"Multi-hop reasoning result: {reasoned_result[:200]}...")
        
        except Exception as e:
            logger.error(f"Error in multi-hop reasoning during reranking: {e}")
    
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

    prompt = PromptTemplate(
        template = template,
        input_variables = ["graph", "question"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(graph = graph, question=question)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(graph = graph, question=question,\
                                                        text={})

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
    """原始函数，使用增强实现"""
    return enhanced_self_knowledge_retrieval_reranking(graph, question)

def enhanced_is_unable_to_answer(response):
    """增强的响应质量验证"""
    if not response or len(response.strip()) < 5:
        return True
    
    failure_patterns = [
        "i don't know", "cannot answer", "insufficient information",
        "unable to determine", "not enough context", "cannot provide"
    ]
    
    response_lower = response.lower()
    for pattern in failure_patterns:
        if pattern in response_lower:
            return True
    
    try:
        analysis = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": response}
            ],
            max_tokens=1,
            temperature=0.0,
            n=1,
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
        return False

def is_unable_to_answer(response):
    """原始函数，使用增强实现"""
    return enhanced_is_unable_to_answer(response)

def enhanced_final_answer(question_text, response_of_KG_list_path, response_of_KG_neighbor):
    """
    增强的最终答案生成函数
    
    功能：整合知识图谱路径信息和邻居信息，使用多策略推理生成医疗问答的最终答案
    
    参数：
    - question_text: 原始医疗问题文本
    - response_of_KG_list_path: 知识图谱路径推理的结果文本
    - response_of_KG_neighbor: 知识图谱邻居实体的结果文本
    
    返回：
    - str: 最终的答案选项（如 "A", "B", "C", "D", "E"）
    """
    
    # ========== 第1步：消融实验配置检查 ==========
    if not ABLATION_CONFIG['USE_ENHANCED_ANSWER_GEN']:
        # 检查全局配置，如果禁用了增强答案生成功能
        logger.info("🔬 Using basic final answer generation")
        # 记录日志，说明使用基础版本
        return basic_final_answer(question_text, response_of_KG_list_path, response_of_KG_neighbor)
        # 直接调用基础版本的答案生成函数并返回
    
    # ========== 第2步：输入数据预处理 ==========
    if response_of_KG_list_path == []:
        # 如果路径推理结果是空列表
        response_of_KG_list_path = ''
        # 将其转换为空字符串，统一数据类型
    if response_of_KG_neighbor == []:
        # 如果邻居推理结果是空列表
        response_of_KG_neighbor = ''
        # 将其转换为空字符串，统一数据类型
    
    # ========== 第3步：问题类型识别和否定词处理 ==========
    question_types = semantic_question_classifier.identify_question_type(question_text)
    # 使用语义问题分类器识别问题类型
    # 例如：['causation']表示因果关系问题，['treatment']表示治疗问题等
    
    has_neg = has_negation(question_text)
    # 检查问题中是否包含否定词（如"not", "except", "excluding"等）
    # 返回布尔值，True表示存在否定，False表示不存在
    
    # ========== 第4步：知识图谱引导推理（KG-guided reasoning）==========
    try:
        kg_subgraph = []
        # 初始化知识子图列表，用于存储三元组[头实体, 关系, 尾实体]
        
        # 处理路径推理结果
        if response_of_KG_list_path:
            # 如果路径推理结果不为空
            path_lines = response_of_KG_list_path.split('\n')
            # 按换行符分割成多行
            for line in path_lines:
                # 遍历每一行
                if '->' in line:
                    # 如果该行包含箭头分隔符（表示实体->关系->实体的格式）
                    parts = line.split('->')
                    # 按箭头分割成不同部分
                    if len(parts) >= 3:
                        # 如果分割后至少有3个部分（头实体、关系、尾实体）
                        kg_subgraph.append([parts[0].strip(), parts[1].strip(), parts[2].strip()])
                        # 去除空白字符并添加到知识子图中
        
        # 处理邻居推理结果
        if response_of_KG_neighbor:
            # 如果邻居推理结果不为空
            neighbor_lines = response_of_KG_neighbor.split('\n')
            # 按换行符分割成多行
            for line in neighbor_lines:
                # 遍历每一行
                if '->' in line:
                    # 如果该行包含箭头分隔符
                    parts = line.split('->')
                    # 按箭头分割
                    if len(parts) >= 3:
                        # 如果分割后至少有3个部分
                        kg_subgraph.append([parts[0].strip(), parts[1].strip(), parts[2].strip()])
                        # 添加到知识子图中
        
        # 执行知识图谱引导推理
        if kg_subgraph and medical_reasoning_rules.kg_guided_reasoning:
            # 如果知识子图不为空且KG引导推理模块可用
            kg_guided_result = medical_reasoning_rules.kg_guided_reasoning.kg_guided_reasoning(
                question_text, kg_subgraph
            )
            # 调用KG引导推理，传入问题文本和知识子图
            logger.debug(f"KG-guided reasoning result: {kg_guided_result[:200]}...")
            # 记录推理结果的前200个字符用于调试
        
    except Exception as e:
        # 如果在KG引导推理过程中出现异常
        logger.error(f"Error in KG-guided reasoning: {e}")
        # 记录错误日志，但不中断程序执行
    
    # ========== 第5步：根据问题类型调整推理指令 ==========
    if has_neg or 'exception' in question_types:
        # 如果问题包含否定词或者问题类型是例外类型
        reasoning_instruction = "Pay special attention to negation words and identify what should be EXCLUDED or what is NOT associated with the topic."
        # 设置否定推理指令，提醒模型注意否定词和排除逻辑
    else:
        # 如果是正常的肯定问题
        reasoning_instruction = "Focus on positive associations and direct relationships."
        # 设置正面推理指令，关注正向关联和直接关系
    
    # ========== 第6步：思维链（Chain-of-Thought）生成 ==========
    messages = [
        # 构建对话消息列表，用于生成思维链推理过程
        SystemMessage(content="You are an excellent AI assistant specialized in medical question answering with access to UMLS standardized medical knowledge and hierarchical reasoning capabilities"),
        # 系统消息：定义AI助手的角色和能力
        HumanMessage(content=f'Question: {question_text}'),
        # 用户消息：提供原始问题
        AIMessage(content=f"You have some medical knowledge information in the following:\n\n" + 
                 f'###Path-based Evidence: {response_of_KG_list_path}\n\n' + 
                 f'###Neighbor-based Evidence: {response_of_KG_neighbor}'),
        # AI消息：提供知识图谱证据，包括路径证据和邻居证据
        HumanMessage(content=f"Answer: Let's think step by step using hierarchical medical reasoning. {reasoning_instruction} ")
        # 用户消息：请求逐步推理，并提供针对问题类型的特定指令
    ]
    
    output_CoT = ""
    # 初始化思维链输出变量
    for retry in range(3):
        # 最多尝试3次生成思维链
        try:
            result_CoT = chat(messages)
            # 调用聊天模型生成思维链推理过程
            if result_CoT.content is not None and len(result_CoT.content.strip()) > 10:
                # 如果生成的内容不为空且长度大于10个字符
                output_CoT = result_CoT.content
                # 保存思维链内容
                break
                # 退出重试循环
            else:
                logger.warning(f"CoT generation attempt {retry + 1} returned insufficient content")
                # 记录警告：内容不足
                sleep(5)
                # 等待5秒后重试
        except Exception as e:
            # 如果生成过程中出现异常
            logger.error(f"CoT generation attempt {retry + 1} failed: {e}")
            # 记录错误日志
            sleep(5)
            # 等待5秒后重试
    
    if not output_CoT:
        # 如果所有重试都失败，思维链为空
        logger.warning("CoT generation failed, using default reasoning")
        # 记录警告日志
        output_CoT = f"Based on the provided medical knowledge, I need to analyze the evidence carefully."
        # 使用默认的推理文本
    
    # ========== 第7步：多次答案生成和投票机制 ==========
    answers = []
    # 初始化答案列表，用于收集多次生成的答案
    for attempt in range(3):
        # 尝试3次生成最终答案
        try:
            final_prompts = [
                # 定义不同的提示词，每次尝试使用不同的提示
                "The final answer (output the letter option) is:",
                "Based on the hierarchical analysis above, the correct answer is:",
                "Therefore, using multi-strategy reasoning, the answer choice is:"
            ]
            
            messages = [
                # 构建最终答案生成的对话消息
                SystemMessage(content="You are an excellent AI assistant specialized in medical question answering with access to UMLS standardized medical knowledge and hierarchical reasoning capabilities"),
                # 系统消息：定义AI助手角色
                HumanMessage(content=f'Question: {question_text}'),
                # 用户消息：原始问题
                AIMessage(content=f"Medical knowledge:\n\n" + 
                         f'###Path-based Evidence: {response_of_KG_list_path}\n\n' + 
                         f'###Neighbor-based Evidence: {response_of_KG_neighbor}'),
                # AI消息：医疗知识证据
                AIMessage(content=f"Analysis: {output_CoT}"),
                # AI消息：前面生成的思维链分析
                AIMessage(content=final_prompts[attempt % len(final_prompts)])
                # AI消息：使用循环方式选择不同的最终提示词
            ]
            
            result = chat(messages)
            # 调用聊天模型生成最终答案
            if result.content is not None and len(result.content.strip()) > 0:
                # 如果生成的内容不为空
                answer_match = re.search(r'\b([A-E])\b', result.content)
                # 使用正则表达式搜索A-E的选项字母
                if answer_match:
                    # 如果找到了匹配的选项字母
                    answers.append(answer_match.group(1))
                    # 将选项字母添加到答案列表
                else:
                    # 如果没有找到标准的选项字母
                    answers.append(result.content.strip()[:10])
                    # 取前10个字符作为答案（备用方案）
                    
        except Exception as e:
            # 如果生成过程中出现异常
            logger.error(f"Final answer attempt {attempt + 1} failed: {e}")
            # 记录错误日志
            sleep(3)
            # 等待3秒后重试
    
    # ========== 第8步：投票选择最终答案 ==========
    if answers:
        # 如果成功生成了至少一个答案
        answer_counts = Counter(answers)
        # 使用Counter统计每个答案出现的次数
        most_common_answer = answer_counts.most_common(1)[0][0]
        # 获取出现次数最多的答案（投票机制）
        
        logger.info(f"Voting results: {dict(answer_counts)}, Selected: {most_common_answer}")
        # 记录投票结果和选择的答案
        return most_common_answer
        # 返回获得最多票数的答案
    
    # ========== 第9步：异常处理 ==========
    logger.error("All final answer attempts failed")
    # 如果所有尝试都失败，记录错误日志
    return "A"
    # 返回默认答案"A"作为兜底方案

def basic_final_answer(question_text, response_of_KG_list_path, response_of_KG_neighbor):
    """基础版本的最终答案生成"""
    if response_of_KG_list_path == []:
        response_of_KG_list_path = ''
    if response_of_KG_neighbor == []:
        response_of_KG_neighbor = ''
    
    messages = [
        SystemMessage(content="You are a medical AI assistant."),
        HumanMessage(content=f'Question: {question_text}'),
        AIMessage(content=f"Knowledge:\n{response_of_KG_list_path}\n{response_of_KG_neighbor}"),
        HumanMessage(content="Answer: The final answer is:")
    ]
    
    try:
        result = chat(messages)
        answer_match = re.search(r'\b([A-E])\b', result.content)
        return answer_match.group(1) if answer_match else "A"
    except:
        return "A"

def final_answer(str, response_of_KG_list_path, response_of_KG_neighbor):
    """原始函数签名保持不变"""
    return enhanced_final_answer(str, response_of_KG_list_path, response_of_KG_neighbor)

def load_and_clean_triples(file_path):
    """从CSV文件加载和清理知识图谱三元组"""
    logger.info("Loading knowledge graph triples...")
    
    df = pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    df_clean = df.dropna().copy()
    
    df_clean.loc[:, 'head'] = df_clean['head'].astype(str).str.strip()
    df_clean.loc[:, 'relation'] = df_clean['relation'].astype(str).str.strip()
    df_clean.loc[:, 'tail'] = df_clean['tail'].astype(str).str.strip()
    
    df_clean = df_clean[(df_clean['head'] != '') & 
                       (df_clean['relation'] != '') & 
                       (df_clean['tail'] != '')]
    
    logger.info(f"Loaded {len(df)} total triples, {len(df_clean)} valid triples after cleaning")
    
    return df_clean

def check_database_populated(session):
    """检查数据库是否已有数据"""
    try:
        result = session.run("MATCH (n) RETURN count(n) as node_count")
        node_count = result.single()["node_count"]
        return node_count > 0
    except:
        return False

import inspect
from functools import wraps

def simple_print_progress(idx, item, step_name, **kwargs):
    """简单直接的进度打印，使用显式参数传递"""
    print(f"\n{'='*80}")
    print(f"\n{step_name}")
    print(f"\n--- Question {idx+1} Progress ---")
    
    # 直接使用传入的参数
    input_text = kwargs.get('input_text', [])
    question_types = kwargs.get('question_types', [])
    question_kg = kwargs.get('question_kg', [])
    match_kg = kwargs.get('match_kg', [])
    confidence_scores = kwargs.get('confidence_scores', [])
    result_path_list = kwargs.get('result_path_list', [])
    neighbor_list = kwargs.get('neighbor_list', [])
    response_of_KG_list_path = kwargs.get('response_of_KG_list_path', '')
    response_of_KG_neighbor = kwargs.get('response_of_KG_neighbor', '')
    output_all = kwargs.get('output_all', '')
    ret_parsed = kwargs.get('ret_parsed', {})
    
    print(f"Question: {input_text[0] if input_text else '未定义'}")
    print(f"Types: {question_types}")
    print(f"Original Entities: {question_kg}")
    
    if match_kg:
        print(f"Matched Entities: {match_kg[:5]}...")
        if confidence_scores:
            print(f"Avg Confidence: {np.mean(confidence_scores):.3f}")
    else:
        print("Matched Entities: 尚未处理")
    
    if result_path_list:
        print(f"Found Paths: {len(result_path_list)}")
    else:
        print("Paths: 尚未查找")
    
    if neighbor_list:
        print(f"Neighbors: {len(neighbor_list)} relations")
    else:
        print("Neighbors: 尚未获取")
    
    if response_of_KG_list_path:
        print(f"Path Response Length: {len(response_of_KG_list_path)}")
    else:
        print("Path Response: 尚未生成")
    
    if response_of_KG_neighbor:
        print(f"Neighbor Response Length: {len(response_of_KG_neighbor)}")
    else:
        print("Neighbor Response: 尚未生成")
    
    if output_all:
        print(f"Final Answer: {output_all}")
        predicted = ret_parsed.get('prediction', '未知') if ret_parsed else '未知'
        correct = item.get('answer', '未知') if item else '未知'
        print(f"Predicted: {predicted}, Correct: {correct}")
    else:
        print("Final Answer: 尚未生成")
    
    print("-" * 50)
    print(f"{'='*80}")

if __name__ == "__main__":
    # 打印当前配置
    # THRESHOLDS.print_config()
    
    # 动态调整阈值（如果需要）
    # THRESHOLDS.set_threshold('entity_matching', 'basic_similarity', 0.65)
    
    # 配置第三方API
    openai.api_key = "sk-P4hNAfoKF4JLckjCuE99XbaN4bZIORZDPllgpwh6PnYWv4cj"
    openai.api_base = "https://aiyjg.lol/v1"
    
    os.environ['OPENAI_API_KEY'] = openai.api_key

    # 1. 构建neo4j知识图谱数据集
    uri = "bolt://localhost:7688"
    username = "neo4j"
    password = "Cyber@511"

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()

        # 检查是否需要重新导入数据
    force_reload = os.getenv('FORCE_RELOAD_DB', 'false').lower() == 'true'
    
    if force_reload or not check_database_populated(session):
        logger.info("Loading knowledge graph data...")
        logger.info("Cleaning existing knowledge graph...")
        session.run("MATCH (n) DETACH DELETE n")

        df_clean = load_and_clean_triples('./Alzheimers/train_s2s.txt')

        batch_size = 1000
        valid_triples = 0
        batch_queries = []
        batch_params = []
        
        logger.info("Starting batch insertion of knowledge graph triples...")
        
        for index, row in tqdm(df_clean.iterrows(), desc="Building knowledge graph"):
            head_name = row['head']
            tail_name = row['tail']
            relation_name = row['relation']
            
            if not validate_knowledge_triple(head_name, relation_name, tail_name):
                continue

            query = (
                "MERGE (h:Entity { name: $head_name }) "
                "MERGE (t:Entity { name: $tail_name }) "
                "MERGE (h)-[r:`" + relation_name + "`]->(t)"
            )
            
            batch_queries.append(query)
            batch_params.append({
                'head_name': head_name,
                'tail_name': tail_name,
                'relation_name': relation_name
            })
            
            if len(batch_queries) >= batch_size:
                try:
                    with driver.session() as batch_session:
                        tx = batch_session.begin_transaction()
                        for q, params in zip(batch_queries, batch_params):
                            tx.run(q, **params)
                        tx.commit()
                    valid_triples += len(batch_queries)
                    logger.debug(f"Successfully inserted batch of {len(batch_queries)} triples")
                except Exception as e:
                    logger.error(f"Failed to insert batch: {e}")
                    for q, params in zip(batch_queries, batch_params):
                        try:
                            session.run(q, **params)
                            valid_triples += 1
                        except Exception as single_e:
                            logger.warning(f"Failed to insert single triple: {params['head_name']} -> {params['relation_name']} -> {params['tail_name']}, Error: {single_e}")
                
                batch_queries = []
                batch_params = []
        
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

    else:
        logger.info("✅ Database already populated, skipping data import")
        logger.info("💡 To force reload, set environment variable: FORCE_RELOAD_DB=true")
        
        # 仍需要加载flat_kg_triples用于层次化图谱构建
        df_clean = load_and_clean_triples('./Alzheimers/train_s2s.txt')


    OPENAI_API_KEY = openai.api_key
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo', temperature=0.7)

    logger.info("Loading embeddings...")
    with open('./Alzheimers/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)
        
    with open('./Alzheimers/keyword_embeddings.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)

    # 新的代码，改进层次化图谱的关键词匹配为语义匹配
    hierarchical_kg_framework = OptimizedHierarchicalKGFramework(
        entity_embeddings=entity_embeddings,
        keyword_embeddings=keyword_embeddings,
        use_semantic_matching=True    # 启用语义匹配
        # similarity_threshold=0.7       # 设置相似度阈值
        # similarity_threshold现在从THRESHOLDS自动获取
    )

    logger.info("Building hierarchical knowledge graph structure...")
    flat_kg_triples = []
    for _, row in df_clean.iterrows():
        flat_kg_triples.append([row['head'], row['relation'], row['tail']])
    
    hierarchical_kg_framework.build_hierarchical_structure(flat_kg_triples)

    # 在加载embeddings之后添加
    logger.info("Initializing semantic question type classifier...")
    semantic_question_classifier = SemanticQuestionTypeClassifier(
        model_name='sentence-transformers/all-mpnet-base-v2',
        # similarity_threshold=0.4
        # similarity_threshold现在从THRESHOLDS自动获取
    )

    # ========================= 初始化增强模块 =========================
    umls_api_key = "7cce913d-29bf-459f-aa9a-2ba57d6efccf"
    umls_normalizer = UMLSNormalizer(umls_api_key)
    medical_reasoning_rules = MedicalReasoningRules(umls_normalizer)
    multi_hop_reasoner = MultiHopReasoning(max_hops=3, umls_normalizer=umls_normalizer)


    medical_reasoning_rules.initialize_kg_guided_reasoning(flat_kg_triples, chat)

    # 只处理第一个数据集
    # datasets = ['medqa', 'medmcqa', 'mmlu', 'qa4mre']
    datasets = ['medqa']  # 只处理第一个数据集

    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset}")
        processor = dataset2processor[dataset]()

        data = processor.load_dataset()

        # ✅ 只取前N个问题
        data = data[:1]

        # ✅ 新增：提取所有问题进行批量处理
        all_questions = []
        for item in data:
            input_text = processor.generate_prompt(item)
            all_questions.append(input_text)
        
        # ✅ 批量处理所有问题类型识别 → 只有1个进度条
        logger.info(f"Batch processing {len(all_questions)} questions for semantic classification...")
        all_question_types = semantic_question_classifier.batch_identify_question_types(all_questions)

        acc, total_num = 0, 0
        generated_data = []

        # ✅ 修改：使用预计算的问题类型
        for idx, item in enumerate(tqdm(data, desc=f"Processing {dataset}")):
            
            if total_num > 0 and total_num % CLEANUP_FREQUENCY == 0:
                cleanup_resources(total_num)

            # 🔧 第一步：立即初始化所有变量（在任何使用之前）
            match_kg = []
            confidence_scores = []
            result_path_list = []
            neighbor_list = []
            response_of_KG_list_path = ""
            response_of_KG_neighbor = ""
            output_all = ""
            ret_parsed = {}
            path_sampled = ""
            neighbor_input_sampled = ""

            input_text = [all_questions[idx]]  # 使用预计算的问题文本
            entity_list = item['entity'].split('\n')
            question_kg = []
            
            for entity in entity_list:
                try:
                    entity = entity.split('.')[1].strip()
                    question_kg.append(entity)
                except:
                    continue


            # ✅ 使用预计算的问题类型，不再重新计算
            question_types = all_question_types[idx]
            logger.info(f"Question types identified: {question_types}")

            # 内容输出
            simple_print_progress(idx, item, "第1步内容打印，使用与计算的问题类型", 
                         input_text=input_text,
                         question_types=question_types,
                         question_kg=question_kg,
                         match_kg=match_kg,  # 空列表
                         confidence_scores=confidence_scores,  # 空列表
                         result_path_list=result_path_list,  # 空列表
                         neighbor_list=neighbor_list,  # 空列表
                         response_of_KG_list_path=response_of_KG_list_path,  # 空字符串
                         response_of_KG_neighbor=response_of_KG_neighbor,  # 空字符串
                         output_all=output_all,  # 空字符串
                         ret_parsed=ret_parsed)  # 空字典

            match_kg, confidence_scores = enhanced_entity_matching(
                question_kg, entity_embeddings, keyword_embeddings, input_text[0])

            # 内容输出
            if idx < 5:  # 只打印前5个的详细信息
                simple_print_progress(idx, item, "第2步内容打印，实体匹配",
                             input_text=input_text,
                             question_types=question_types,
                             question_kg=question_kg,
                             match_kg=match_kg,  # 现在有内容
                             confidence_scores=confidence_scores,  # 现在有内容
                             result_path_list=result_path_list,  # 仍然空
                             neighbor_list=neighbor_list,  # 仍然空
                             response_of_KG_list_path=response_of_KG_list_path,
                             response_of_KG_neighbor=response_of_KG_neighbor,
                             output_all=output_all,
                             ret_parsed=ret_parsed)
            

            if len(match_kg) < 2:
                logger.warning(f"Insufficient entities matched for question: {input_text[0][:100]}...")
                match_kg.extend(question_kg[:2])

            # 内容输出
            if idx < 5:  # 只打印前5个的详细信息
                simple_print_progress(idx, item, "第3步内容打印，知识图谱路径查找之前",
                             input_text=input_text,
                             question_types=question_types,
                             question_kg=question_kg,
                             match_kg=match_kg,  # 现在有内容
                             confidence_scores=confidence_scores,  # 现在有内容
                             result_path_list=result_path_list,  # 仍然空
                             neighbor_list=neighbor_list,  # 仍然空
                             response_of_KG_list_path=response_of_KG_list_path,
                             response_of_KG_neighbor=response_of_KG_neighbor,
                             output_all=output_all,
                             ret_parsed=ret_parsed)

            # 4. 增强的neo4j知识图谱路径查找，查找初始实体和候选实体之间的最优路径
            if len(match_kg) > 1:
                start_entity = match_kg[0]
                candidate_entity = match_kg[1:]
                
                result_path_list = []
                while True:
                    flag = 0
                    paths_list = []
                    
                    while candidate_entity:
                        end_entity = candidate_entity[0]
                        candidate_entity.remove(end_entity)
                        
                        paths, exist_entity = find_shortest_path(start_entity, end_entity, candidate_entity, question_types)
                        path_list = []
                        
                        if paths == [''] or paths == []:
                            flag = 1
                            if not candidate_entity:
                                flag = 0
                                break
                            start_entity = candidate_entity[0]
                            candidate_entity.remove(start_entity)
                            break
                        else:
                            for p in paths:
                                path_list.append(p.split('->'))
                            if path_list:
                                paths_list.append(path_list)
                        
                        if exist_entity != {}:
                            try:
                                candidate_entity.remove(exist_entity)
                            except:
                                continue
                        start_entity = end_entity
                    
                    result_path = combine_lists(*paths_list)
                    
                    if result_path:
                        result_path_list.extend(result_path)
                    if flag == 1:
                        continue
                    else:
                        break
                        
                start_tmp = []
                for path_new in result_path_list:
                    if path_new == []:
                        continue
                    if path_new[0] not in start_tmp:
                        start_tmp.append(path_new[0])
                
                if len(start_tmp) == 0:
                    result_path = {}
                    single_path = {}
                else:
                    if len(start_tmp) == 1:
                        result_path = result_path_list[:5]
                    else:
                        result_path = []
                                                                
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

            # 内容输出
            if idx < 5:  # 只打印前5个的详细信息
                simple_print_progress(idx, item, "第4步内容打印，增强的neo4j知识图谱路径查找",
                             input_text=input_text,
                             question_types=question_types,
                             question_kg=question_kg,
                             match_kg=match_kg,
                             confidence_scores=confidence_scores,
                             result_path_list=result_path_list,
                             neighbor_list=neighbor_list,
                             response_of_KG_list_path=response_of_KG_list_path,
                             response_of_KG_neighbor=response_of_KG_neighbor,
                             output_all=output_all,
                             ret_parsed=ret_parsed)
            
            # 5. 增强的neo4j知识图谱邻居实体,获取邻居信息
            neighbor_list = []
            neighbor_list_disease = []
            
            for match_entity in match_kg:
                disease_flag = 0
                neighbors, disease = get_entity_neighbors(match_entity, disease_flag, question_types)
                neighbor_list.extend(neighbors)

                try:
                    hierarchical_context = hierarchical_kg_framework.get_hierarchical_context(
                        match_entity, context_type='all'
                    )
                    
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
            
            if len(neighbor_list) <= 5:
                neighbor_list.extend(neighbor_list_disease)

            # 内容输出
            if idx < 5:  # 只打印前5个的详细信息
                simple_print_progress(idx, item, "第5步内容打印，增强的neo4j知识图谱邻居实体",
                             input_text=input_text,
                             question_types=question_types,
                             question_kg=question_kg,
                             match_kg=match_kg,
                             confidence_scores=confidence_scores,
                             result_path_list=result_path_list,
                             neighbor_list=neighbor_list,
                             response_of_KG_list_path=response_of_KG_list_path,
                             response_of_KG_neighbor=response_of_KG_neighbor,
                             output_all=output_all,
                             ret_parsed=ret_parsed)

            # 6. 增强的知识图谱路径基础提示生成，将路径转换成自然语言提示
            if len(match_kg) > 1:
                response_of_KG_list_path = []
                if result_path == {}:
                    response_of_KG_list_path = []
                    path_sampled = []
                else:
                    result_new_path = []
                    for total_path_i in result_path:
                        path_input = "->".join(total_path_i)
                        result_new_path.append(path_input)
                    
                    path = "\n".join(result_new_path)
                    path_sampled = self_knowledge_retrieval_reranking(path, input_text[0])
                    response_of_KG_list_path = prompt_path_finding(path_sampled)
                    
                    # 重试屏蔽位置
                    if is_unable_to_answer(response_of_KG_list_path):
                        logger.warning("Path finding response validation failed, retrying...")
                        response_of_KG_list_path = prompt_path_finding(path_sampled)
                    # 结束位置
            else:
                response_of_KG_list_path = '{}'

            try:
                response_single_path = prompt_path_finding(single_path)
                if is_unable_to_answer(response_single_path):
                    response_single_path = prompt_path_finding(single_path)
            except:
                response_single_path = ""

            # 内容输出
            if idx < 5:  # 只打印前5个的详细信息
                simple_print_progress(idx, item, "第6步内容打印，增强的知识图谱路径基础提示生成",
                             input_text=input_text,
                             question_types=question_types,
                             question_kg=question_kg,
                             match_kg=match_kg,
                             confidence_scores=confidence_scores,
                             result_path_list=result_path_list,
                             neighbor_list=neighbor_list,
                             response_of_KG_list_path=response_of_KG_list_path,
                             response_of_KG_neighbor=response_of_KG_neighbor,
                             output_all=output_all,
                             ret_parsed=ret_parsed)

            # 7. 增强的知识图谱邻居实体基础提示生成
            response_of_KG_list_neighbor = []
            neighbor_new_list = []
            
            for neighbor_i in neighbor_list:
                neighbor = "->".join(neighbor_i)
                neighbor_new_list.append(neighbor)

            if len(neighbor_new_list) > 5:
                neighbor_input = "\n".join(neighbor_new_list[:5])
            else:
                neighbor_input = "\n".join(neighbor_new_list)
            
            neighbor_input_sampled = self_knowledge_retrieval_reranking(neighbor_input, input_text[0])
            response_of_KG_neighbor = prompt_neighbor(neighbor_input_sampled)
            
            # 重试屏蔽位置
            if is_unable_to_answer(response_of_KG_neighbor):
                logger.warning("Neighbor processing response validation failed, retrying...")
                response_of_KG_neighbor = prompt_neighbor(neighbor_input_sampled)
            # 结束位置

            # 内容输出
            if idx < 5:  # 只打印前5个的详细信息
                simple_print_progress(idx, item, "第7步内容打印，增强的知识图谱邻居实体基础提示生成",
                             input_text=input_text,
                             question_types=question_types,
                             question_kg=question_kg,
                             match_kg=match_kg,
                             confidence_scores=confidence_scores,
                             result_path_list=result_path_list,
                             neighbor_list=neighbor_list,
                             response_of_KG_list_path=response_of_KG_list_path,
                             response_of_KG_neighbor=response_of_KG_neighbor,
                             output_all=output_all,
                             ret_parsed=ret_parsed)

            # 8. 增强的基于提示的医学对话答案生成（移除了置信度计算）
            output_all = enhanced_final_answer(input_text[0], response_of_KG_list_path, response_of_KG_neighbor)
            
            # 重试屏蔽位置
            if is_unable_to_answer(output_all):
                logger.warning("Final answer validation failed, retrying...")
                output_all = enhanced_final_answer(input_text[0], response_of_KG_list_path, response_of_KG_neighbor)
            # 结束位置

            # 内容输出
            if idx < 5:  # 只打印前5个的详细信息
                simple_print_progress(idx, item, "第8步内容打印，增强的基于提示的医学对话答案生成（移除了置信度计算）",
                             input_text=input_text,
                             question_types=question_types,
                             question_kg=question_kg,
                             match_kg=match_kg,
                             confidence_scores=confidence_scores,
                             result_path_list=result_path_list,
                             neighbor_list=neighbor_list,
                             response_of_KG_list_path=response_of_KG_list_path,
                             response_of_KG_neighbor=response_of_KG_neighbor,
                             output_all=output_all,
                             ret_parsed=ret_parsed)

            ret_parsed, acc_item = processor.parse(output_all, item)
            ret_parsed['path'] = path_sampled if 'path_sampled' in locals() else ""
            ret_parsed['neighbor_input'] = neighbor_input_sampled if 'neighbor_input_sampled' in locals() else ""
            ret_parsed['response_of_KG_list_path'] = response_of_KG_list_path
            ret_parsed['response_of_KG_neighbor'] = response_of_KG_neighbor
            ret_parsed['entity_confidence_scores'] = confidence_scores if 'confidence_scores' in locals() else []
            ret_parsed['question_types'] = question_types
            
            try:
                ret_parsed['umls_normalized_entities'] = umls_normalizer.normalize_medical_terms(question_kg)
                ret_parsed['umls_semantic_variants'] = [umls_normalizer.get_semantic_variants(entity)[:3] for entity in question_kg[:3]]
                
                enhanced_links = umls_normalizer.enhanced_entity_linking_method(
                    question_kg, input_text[0], question_types
                )
                ret_parsed['enhanced_entity_links'] = enhanced_links
                
                adaptive_knowledge = umls_normalizer.adaptive_knowledge_selection(
                    question_types, question_kg
                )
                ret_parsed['adaptive_knowledge_count'] = len(adaptive_knowledge)
                
                hierarchical_contexts = {}
                for entity in question_kg[:3]:
                    hierarchical_contexts[entity] = hierarchical_kg_framework.get_hierarchical_context(
                        entity, context_type='all'
                    )
                ret_parsed['hierarchical_contexts'] = hierarchical_contexts
                
                if len(question_kg) >= 2:
                    multi_hop_paths = multi_hop_reasoner.optimized_multi_hop.intelligent_path_selection(
                        question_kg[:1], question_kg[1:2], max_hops=2
                    )
                    ret_parsed['multi_hop_paths_count'] = len(multi_hop_paths)
                else:
                    ret_parsed['multi_hop_paths_count'] = 0
                
            except Exception as e:
                logger.error(f"Error in enhanced processing: {e}")
                ret_parsed['umls_normalized_entities'] = question_kg
                ret_parsed['umls_semantic_variants'] = []
                ret_parsed['enhanced_entity_links'] = {}
                ret_parsed['adaptive_knowledge_count'] = 0
                ret_parsed['hierarchical_contexts'] = {}
                ret_parsed['multi_hop_paths_count'] = 0
            
            ret_parsed = convert_numpy_types(ret_parsed)
            
            if ret_parsed['prediction'] in processor.num2answer.values():
                acc += acc_item
                total_num += 1
            generated_data.append(ret_parsed)

        logger.info(f"Dataset: {dataset}")
        logger.info(f"Accuracy: {acc/total_num:.4f} ({acc}/{total_num})")

        # 内容输出
        if idx < 5:  # 只打印前5个的详细信息
            simple_print_progress(idx, item, "第9步内容打印，后续处理工作完成",
                            input_text=input_text,
                            question_types=question_types,
                            question_kg=question_kg,
                            match_kg=match_kg,
                            confidence_scores=confidence_scores,
                            result_path_list=result_path_list,
                            neighbor_list=neighbor_list,
                            response_of_KG_list_path=response_of_KG_list_path,
                            response_of_KG_neighbor=response_of_KG_neighbor,
                            output_all=output_all,
                            ret_parsed=ret_parsed)

        os.makedirs('./Alzheimers/result_chatgpt_mindmap', exist_ok=True)
        
        output_filename = f"{dataset}_{CURRENT_ABLATION_CONFIG}_ablation_results.json"
        with open(os.path.join('./Alzheimers/result_chatgpt_mindmap', output_filename), 'w') as f:
            json.dump(generated_data, fp=f, indent=2)
            
        logger.info(f"Ablation results saved for dataset: {dataset}")
        
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
        
        performance_stats['question_type_distribution'] = question_type_counts
        performance_stats['hierarchical_context_coverage'] = hierarchical_coverage_count / len(generated_data) if generated_data else 0
        performance_stats['multi_strategy_usage'] = multi_strategy_count / len(generated_data) if generated_data else 0
        
        stats_filename = f"{dataset}_{CURRENT_ABLATION_CONFIG}_performance_stats.json"
        with open(os.path.join('./Alzheimers/result_chatgpt_mindmap', stats_filename), 'w') as f:
            json.dump(performance_stats, fp=f, indent=2)
            
        logger.info(f"Performance statistics saved for dataset: {dataset}")
        logger.info(f"Hierarchical context coverage: {performance_stats['hierarchical_context_coverage']:.3f}")
        logger.info(f"Multi-strategy usage: {performance_stats['multi_strategy_usage']:.3f}")

    logger.info("="*50)
    logger.info(f"🎉 Ablation study completed for configuration: {CURRENT_ABLATION_CONFIG}")
    logger.info("📊 Ablation configuration applied:")
    for module, enabled in ABLATION_CONFIG.items():
        status = "✅ ENABLED" if enabled else "❌ DISABLED"
        logger.info(f"   {module}: {status}")
    logger.info("="*50)
    
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
    
    with open('./Alzheimers/result_chatgpt_mindmap/ablation_experiment_report.json', 'w') as f:
        json.dump(overall_stats, fp=f, indent=2)
    
    logger.info("📈 Ablation experiment report saved!")
    
    driver.close()
    
    logger.info("🔌 Database connection closed. Ablation study complete!")
    logger.info(f"🔬 To run different ablation configurations, set ABLATION_CONFIG environment variable to one of: {list(ABLATION_CONFIGS.keys())}")

