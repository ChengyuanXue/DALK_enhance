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
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from langchain.llms import OpenAI
import os
from PIL import Image, ImageDraw, ImageFont
import csv
from gensim import corpora
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from rank_bm25 import BM25Okapi
from gensim.models import Word2Vec
import sys
from time import sleep
import logging
from functools import wraps
import requests
import urllib.parse
import hashlib
import hmac
import base64
from datetime import datetime
import xml.etree.ElementTree as ET
import gc

from dataset_utils import *
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            
        self.similarity_threshold = similarity_threshold
        
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

# ========================= 层次化知识图谱架构 =========================

class HierarchicalKGFramework:
    def __init__(self):
        """初始化层次化知识图谱框架"""
        self.disease_hierarchy = defaultdict(list)
        self.symptom_hierarchy = defaultdict(list)
        self.treatment_hierarchy = defaultdict(list)
        # self.anatomy_hierarchy = defaultdict(list)
        # self.pathology_hierarchy = defaultdict(list)
        
        self.hierarchy_weights = {
            'is_a': 1.0,
            'part_of': 0.9,
            'subtype_of': 0.95,
            'category_of': 0.8,
            'related_to': 0.6
        }
    
    def build_hierarchical_structure(self, flat_kg):
        """构建层次化知识结构"""
        if not ABLATION_CONFIG['USE_HIERARCHICAL_KG']:
            logger.info("🔬 Hierarchical KG Framework disabled in ablation study")
            return
            
        logger.info("Building hierarchical knowledge structure...")
        
        self._build_disease_hierarchy(flat_kg)
        self._build_symptom_hierarchy(flat_kg)
        self._build_treatment_hierarchy(flat_kg)
        # self._build_anatomy_hierarchy(flat_kg)
        
        logger.info(f"Built hierarchies: diseases={len(self.disease_hierarchy)}, "
                   f"symptoms={len(self.symptom_hierarchy)}, "
                   f"treatments={len(self.treatment_hierarchy)}")
    
    def _build_disease_hierarchy(self, flat_kg):
        """构建疾病分类层次"""
        for triple in flat_kg:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                if any(keyword in relation.lower() for keyword in 
                       ['is_a', 'subtype', 'category', 'type_of']):
                    if any(keyword in head.lower() for keyword in 
                           ['disease', 'syndrome', 'disorder', 'condition']):
                        self.disease_hierarchy[tail].append({
                            'entity': head,
                            'relation': relation,
                            'weight': self.hierarchy_weights.get(relation.lower(), 0.5)
                        })
    
    def _build_symptom_hierarchy(self, flat_kg):
        """构建症状-疾病关联层次"""
        for triple in flat_kg:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                if any(keyword in relation.lower() for keyword in 
                       ['symptom', 'sign', 'manifestation', 'presents']):
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
                
                if any(keyword in relation.lower() for keyword in 
                       ['treat', 'therapy', 'medication', 'drug']):
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
                
                if any(keyword in relation.lower() for keyword in 
                       ['part_of', 'located_in', 'contains', 'anatomy']):
                    self.anatomy_hierarchy[tail].append({
                        'entity': head,
                        'relation': relation,
                        'weight': self.hierarchy_weights.get(relation.lower(), 0.6)
                    })
    
    def get_hierarchical_context(self, entity, context_type='all'):
        """获取实体的层次化上下文"""
        if not ABLATION_CONFIG['USE_HIERARCHICAL_KG']:
            return {}
            
        context = {}
        
        if context_type in ['all', 'disease']:
            context['diseases'] = self.disease_hierarchy.get(entity, [])
        
        if context_type in ['all', 'symptom']:
            context['symptoms'] = self.symptom_hierarchy.get(entity, [])
        
        if context_type in ['all', 'treatment']:
            context['treatments'] = self.treatment_hierarchy.get(entity, [])
        
        # if context_type in ['all', 'anatomy']:
        #     context['anatomy'] = self.anatomy_hierarchy.get(entity, [])
        
        return context


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
        self.thresholds = {
            'disease': 0.65,
            'symptom': 0.60, 
            'treatment': 0.58
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


# ========================= 层次化图谱构建_优化的层次构建类 =========================

class OptimizedHierarchicalKGFramework:
    def __init__(self, entity_embeddings, keyword_embeddings, use_semantic_matching=True, similarity_threshold=0.7):
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
        self.semantic_matcher = OptimizedSemanticMatcher(self.vector_manager)
        self.use_semantic_matching = use_semantic_matching
    
    def build_hierarchical_structure(self, flat_kg):
        """优化的层次结构构建"""
        if not ABLATION_CONFIG['USE_HIERARCHICAL_KG']:
            logger.info("Hierarchical KG Framework disabled in ablation study")
            return
            
        logger.info("Building optimized hierarchical knowledge structure...")
        
        # 批量分类和构建
        self._build_hierarchy_optimized(flat_kg, 'disease')
        self._build_hierarchy_optimized(flat_kg, 'symptom') 
        self._build_hierarchy_optimized(flat_kg, 'treatment')
        
        logger.info(f"Built optimized hierarchies: diseases={len(self.disease_hierarchy)}, "
                   f"symptoms={len(self.symptom_hierarchy)}, treatments={len(self.treatment_hierarchy)}")
    
    def _build_hierarchy_optimized(self, flat_kg, concept_type):
        """优化的单个层次构建"""
        # 批量语义分类
        matched_triples = self.semantic_matcher.classify_triple_batch(flat_kg, concept_type)
        
        hierarchy_dict = getattr(self, f"{concept_type}_hierarchy")
        
        progress_desc = f"Building {concept_type} hierarchy"
        for match_info in tqdm(matched_triples, desc=progress_desc, miniters=100):
            triple = match_info['triple']
            score = match_info['score']
            
            head, relation, tail = triple[0], triple[1], triple[2]
            
            # 根据概念类型确定层次结构
            if concept_type == 'disease':
                # 疾病层次：子类 -> 父类
                hierarchy_dict[tail].append({
                    'entity': head,
                    'relation': relation,
                    'weight': self.hierarchy_weights.get(relation.lower(), 0.5),
                    'semantic_score': score
                })
            else:
                # 症状/治疗层次：实体 -> 相关项
                hierarchy_dict[head].append({
                    'entity': tail, 
                    'relation': relation,
                    'weight': self.hierarchy_weights.get(relation.lower(), 0.7),
                    'semantic_score': score
                })
        
        # 按语义得分排序
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


# ========================= 改进的层次化知识图谱架构 =========================


class EnhancedHierarchicalKGFramework:
    """
    增强的层次化知识图谱框架类
    结合关键词匹配和语义相似度匹配
    """
    def __init__(self, use_semantic_matching=True, similarity_threshold=0.7):
        """初始化层次化知识图谱框架"""
        # 保持原有的数据结构
        self.disease_hierarchy = defaultdict(list)
        self.symptom_hierarchy = defaultdict(list)
        self.treatment_hierarchy = defaultdict(list)
        # self.anatomy_hierarchy = defaultdict(list)
        # self.pathology_hierarchy = defaultdict(list)
        
        # 保持原有的权重设置
        self.hierarchy_weights = {
            'is_a': 1.0,
            'part_of': 0.9,
            'subtype_of': 0.95,
            'category_of': 0.8,
            'related_to': 0.6
        }
        
        # 新增：语义匹配配置
        self.use_semantic_matching = use_semantic_matching
        self.similarity_threshold = similarity_threshold
        self.semantic_model = None
        
        # 初始化语义模型（如果启用）
        if self.use_semantic_matching:
            try:
                self.semantic_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                logger.info("✅ Semantic similarity model loaded successfully")
                
                # 预定义的关键词嵌入（一次性计算，提高效率）
                self._precompute_keyword_embeddings()
                
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}. Falling back to keyword matching.")
                self.use_semantic_matching = False
    
    def _precompute_keyword_embeddings(self):
        """预计算关键词嵌入向量"""
        # 疾病相关关键词
        self.disease_keywords = ['is_a', 'subtype', 'category', 'type_of', 'disease', 'syndrome', 'disorder', 'condition']
        
        # 症状相关关键词
        self.symptom_keywords = ['symptom', 'sign', 'manifestation', 'presents', 'shows', 'exhibits']
        
        # 治疗相关关键词
        self.treatment_keywords = ['treat', 'therapy', 'medication', 'drug', 'cure', 'heal', 'remedy']
        
        # 解剖相关关键词
        self.anatomy_keywords = ['part_of', 'located_in', 'contains', 'anatomy', 'structure', 'organ']
        
        # 计算嵌入向量
        if self.semantic_model:
            self.disease_embeddings = self.semantic_model.encode(self.disease_keywords)
            self.symptom_embeddings = self.semantic_model.encode(self.symptom_keywords)
            self.treatment_embeddings = self.semantic_model.encode(self.treatment_keywords)
            self.anatomy_embeddings = self.semantic_model.encode(self.anatomy_keywords)
    
    def _semantic_similarity_check(self, text, keyword_embeddings, keyword_type="general"):
        """
        检查文本与关键词的语义相似度
        输入: text - 待检查文本, keyword_embeddings - 关键词嵌入, keyword_type - 关键词类型
        返回: (是否匹配, 最高相似度分数)
        """
        if not self.use_semantic_matching or not self.semantic_model:
            return False, 0.0
        
        try:
            # 计算文本嵌入
            text_embedding = self.semantic_model.encode([text], show_progress_bar=False)  # 禁用进度条
            
            # 计算与所有关键词的相似度
            similarities = cosine_similarity(text_embedding, keyword_embeddings)[0]
            max_similarity = np.max(similarities)
            
            # 根据关键词类型调整阈值
            threshold = self.similarity_threshold
            if keyword_type == "disease":
                threshold = 0.6  # 疾病相关可以稍微宽松
            elif keyword_type == "treatment":
                threshold = 0.65  # 治疗相关中等严格
            
            is_match = max_similarity >= threshold
            
            if is_match:
                best_keyword_idx = np.argmax(similarities)
                logger.debug(f"Semantic match: '{text}' ~ '{keyword_type}' (similarity: {max_similarity:.3f})")
            
            return is_match, max_similarity
            
        except Exception as e:
            logger.error(f"Error in semantic similarity check: {e}")
            return False, 0.0
    
    def _enhanced_keyword_check(self, relation, head_entity, keyword_list, keyword_embeddings, keyword_type):
        """
        增强的关键词检查，结合传统匹配和语义匹配
        输入: relation - 关系, head_entity - 头实体, keyword_list - 关键词列表, 
             keyword_embeddings - 关键词嵌入, keyword_type - 关键词类型
        返回: 是否匹配
        """
        # 首先尝试传统关键词匹配（速度快）
        traditional_match = any(keyword in relation.lower() for keyword in keyword_list)
        
        if traditional_match:
            return True
        
        # 如果传统匹配失败，尝试语义匹配
        if self.use_semantic_matching:
            # 检查关系的语义匹配
            relation_match, _ = self._semantic_similarity_check(relation, keyword_embeddings, keyword_type)
            
            # 检查头实体的语义匹配（补充信息）
            entity_match, _ = self._semantic_similarity_check(head_entity, keyword_embeddings, keyword_type)
            
            # 任一匹配即可
            return relation_match or entity_match
        
        return False
    
    def build_hierarchical_structure(self, flat_kg):
        """
        构建层次化知识结构（保持原有接口）
        输入: flat_kg - 平坦的知识图谱三元组列表
        """
        if not ABLATION_CONFIG['USE_HIERARCHICAL_KG']:
            logger.info("🔬 Hierarchical KG Framework disabled in ablation study")
            return
            
        method_info = "enhanced semantic + keyword matching" if self.use_semantic_matching else "keyword matching only"
        logger.info(f"Building hierarchical knowledge structure using {method_info}...")
        
        # 调用增强的构建方法
        self._build_disease_hierarchy_enhanced(flat_kg)
        self._build_symptom_hierarchy_enhanced(flat_kg)
        self._build_treatment_hierarchy_enhanced(flat_kg)
        # self._build_anatomy_hierarchy_enhanced(flat_kg)
        
        # 记录构建结果
        logger.info(f"Built enhanced hierarchies: diseases={len(self.disease_hierarchy)}, "
                   f"symptoms={len(self.symptom_hierarchy)}, "
                   f"treatments={len(self.treatment_hierarchy)}")
    
    def _build_disease_hierarchy_enhanced(self, flat_kg):
        """增强的疾病分类层次构建"""
        keyword_list = ['is_a', 'subtype', 'category', 'type_of']
        entity_keywords = ['disease', 'syndrome', 'disorder', 'condition']
        
        for triple in tqdm(flat_kg, desc="Building disease hierarchy", miniters=1000, mininterval=30):
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                # 使用增强的关键词检查
                relation_match = self._enhanced_keyword_check(
                    relation, head, keyword_list, 
                    self.disease_embeddings if self.use_semantic_matching else None, 
                    "disease"
                )
                
                entity_match = self._enhanced_keyword_check(
                    head, head, entity_keywords,
                    self.disease_embeddings if self.use_semantic_matching else None,
                    "disease"
                )
                
                if relation_match and entity_match:
                    self.disease_hierarchy[tail].append({
                        'entity': head,
                        'relation': relation,
                        'weight': self.hierarchy_weights.get(relation.lower(), 0.5)
                    })
    
    def _build_symptom_hierarchy_enhanced(self, flat_kg):
        """增强的症状-疾病关联层次构建"""
        keyword_list = ['symptom', 'sign', 'manifestation', 'presents']
        
        for triple in tqdm(flat_kg, desc="Building symptom hierarchy", miniters=1000, mininterval=30):
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                relation_match = self._enhanced_keyword_check(
                    relation, head, keyword_list,
                    self.symptom_embeddings if self.use_semantic_matching else None,
                    "symptom"
                )
                
                if relation_match:
                    self.symptom_hierarchy[head].append({
                        'entity': tail,
                        'relation': relation,
                        'weight': self.hierarchy_weights.get(relation.lower(), 0.7)
                    })
    
    def _build_treatment_hierarchy_enhanced(self, flat_kg):
        """增强的治疗方案层次构建"""
        keyword_list = ['treat', 'therapy', 'medication', 'drug']
        
        for triple in tqdm(flat_kg, desc="Building treatment hierarchy", miniters=1000, mininterval=30):
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                relation_match = self._enhanced_keyword_check(
                    relation, head, keyword_list,
                    self.treatment_embeddings if self.use_semantic_matching else None,
                    "treatment"
                )
                
                if relation_match:
                    self.treatment_hierarchy[head].append({
                        'entity': tail,
                        'relation': relation,
                        'weight': self.hierarchy_weights.get(relation.lower(), 0.8)
                    })
    
    def _build_anatomy_hierarchy_enhanced(self, flat_kg):
        """增强的解剖结构层次构建"""
        keyword_list = ['part_of', 'located_in', 'contains', 'anatomy']
        
        for triple in flat_kg:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                relation_match = self._enhanced_keyword_check(
                    relation, head, keyword_list,
                    self.anatomy_embeddings if self.use_semantic_matching else None,
                    "anatomy"
                )
                
                if relation_match:
                    self.anatomy_hierarchy[tail].append({
                        'entity': head,
                        'relation': relation,
                        'weight': self.hierarchy_weights.get(relation.lower(), 0.6)
                    })
    
    def get_hierarchical_context(self, entity, context_type='all'):
        """
        获取实体的层次化上下文（保持原有接口）
        输入: entity - 实体名称, context_type - 上下文类型
        返回: 包含不同层次上下文的字典
        """
        if not ABLATION_CONFIG['USE_HIERARCHICAL_KG']:
            return {}
            
        context = {}
        
        if context_type in ['all', 'disease']:
            context['diseases'] = self.disease_hierarchy.get(entity, [])
        
        if context_type in ['all', 'symptom']:
            context['symptoms'] = self.symptom_hierarchy.get(entity, [])
        
        if context_type in ['all', 'treatment']:
            context['treatments'] = self.treatment_hierarchy.get(entity, [])
        
        # if context_type in ['all', 'anatomy']:
        #     context['anatomy'] = self.anatomy_hierarchy.get(entity, [])
        
        return context


# ========================= 多策略实体链接 =========================

class SemanticMatcher:
    def __init__(self):
        self.similarity_threshold = 0.7
    
    def match(self, entities, umls_kg):
        """语义相似度匹配"""
        if not ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
            return {}
            
        matches = {}
        for entity in entities:
            best_match = None
            best_score = 0
            
            for kg_entity in umls_kg:
                score = self._calculate_semantic_similarity(entity, kg_entity)
                if score > best_score and score > self.similarity_threshold:
                    best_score = score
                    best_match = kg_entity
            
            if best_match:
                matches[entity] = {'match': best_match, 'score': best_score, 'method': 'semantic'}
        
        return matches
    
    def _calculate_semantic_similarity(self, entity1, entity2):
        """计算语义相似度"""
        words1 = set(entity1.lower().split())
        words2 = set(entity2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0

class ContextAwareLinker:
    def __init__(self):
        self.context_weight = 0.3
    
    def link(self, entities, context):
        """上下文感知链接"""
        if not ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
            return {}
            
        links = {}
        context_words = set(context.lower().split())
        
        for entity in entities:
            entity_words = set(entity.lower().split())
            context_overlap = len(entity_words.intersection(context_words))
            context_score = context_overlap / len(entity_words) if entity_words else 0
            
            links[entity] = {
                'context_score': context_score,
                'method': 'context_aware'
            }
        
        return links

class ConfidenceEstimator:
    def __init__(self):
        self.weight_semantic = 0.6
        self.weight_context = 0.4
    
    def fuse_results(self, semantic_matches, context_matches):
        """置信度估计和融合"""
        if not ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
            return {}
            
        final_links = {}
        
        all_entities = set(semantic_matches.keys()) | set(context_matches.keys())
        
        for entity in all_entities:
            semantic_score = semantic_matches.get(entity, {}).get('score', 0)
            context_score = context_matches.get(entity, {}).get('context_score', 0)
            
            combined_score = (self.weight_semantic * semantic_score + 
                            self.weight_context * context_score)
            
            final_links[entity] = {
                'final_score': combined_score,
                'semantic_score': semantic_score,
                'context_score': context_score,
                'method': 'fused'
            }
        
        return final_links

class EnhancedEntityLinking:
    def __init__(self):
        self.semantic_matcher = SemanticMatcher()
        self.context_aware_linker = ContextAwareLinker()
        self.confidence_estimator = ConfidenceEstimator()
    
    def multi_strategy_linking(self, entities, context, umls_kg):
        """多策略实体链接"""
        if not ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
            return {}
            
        semantic_matches = self.semantic_matcher.match(entities, umls_kg)
        context_matches = self.context_aware_linker.link(entities, context)
        final_links = self.confidence_estimator.fuse_results(
            semantic_matches, context_matches
        )
        
        return final_links

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
            concepts = self.umls_api.search_concepts(entity)
            if concepts and 'results' in concepts:
                for concept in concepts['results'][:3]:
                    cui = concept.get('ui', '')
                    relations = self.umls_api.get_concept_relations(cui)
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
        self.umls_api = UMLS_API(api_key)
        self.local_cache = {}
        self.semantic_type_cache = {}
        
        self.hierarchical_kg = HierarchicalKGFramework()
        self.enhanced_entity_linking = EnhancedEntityLinking()
        self.adaptive_umls_selector = AdaptiveUMLSSelector(self.umls_api)
        
        self.semantic_type_priority = {
            'T047': 10,  # Disease or Syndrome
            'T184': 9,   # Sign or Symptom
            'T061': 8,   # Therapeutic or Preventive Procedure
            'T121': 7,   # Pharmacologic Substance
            'T023': 6,   # Body Part, Organ, or Organ Component
            'T037': 5,   # Injury or Poisoning
            'T046': 4,   # Pathologic Function
            'T033': 3,   # Finding
            'T170': 2,   # Intellectual Product
            'T169': 1    # Functional Concept
        }
    
    def _get_best_cui_for_term(self, term):
        """为给定术语获取最佳CUI"""
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return None
            
        if term in self.local_cache:
            return self.local_cache[term]
        
        try:
            search_results = self.umls_api.search_concepts(term)
            
            if not search_results or 'results' not in search_results:
                return None
            
            results = search_results['results']
            if not results:
                return None
            
            best_cui = None
            best_score = 0
            
            for result in results:
                cui = result['ui']
                name = result['name']
                
                score = self._calculate_match_score(term, name, result)
                
                if score > best_score:
                    best_score = score
                    best_cui = cui
            
            self.local_cache[term] = best_cui
            return best_cui
            
        except Exception as e:
            logger.error(f"Error getting CUI for term '{term}': {e}")
            return None
    
    def _calculate_match_score(self, original_term, concept_name, result):
        """计算匹配分数"""
        score = 0
        
        if original_term.lower() == concept_name.lower():
            score += 100
        elif original_term.lower() in concept_name.lower():
            score += 50
        elif concept_name.lower() in original_term.lower():
            score += 30
        
        original_words = set(original_term.lower().split())
        concept_words = set(concept_name.lower().split())
        overlap = len(original_words & concept_words)
        score += overlap * 10
        
        if self._has_root_match(original_term, concept_name):
            score += 20
        
        return score
    
    def _has_root_match(self, term1, term2):
        """检查词根匹配"""
        suffixes = ['s', 'es', 'ing', 'ed', 'er', 'est', 'ly']
        
        def get_root(word):
            for suffix in suffixes:
                if word.endswith(suffix):
                    return word[:-len(suffix)]
            return word
        
        root1 = get_root(term1.lower())
        root2 = get_root(term2.lower())
        
        return root1 == root2 or root1 in root2 or root2 in root1
    
    def get_concept_synonyms(self, cui):
        """获取概念的同义词"""
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return []
            
        try:
            atoms_result = self.umls_api.get_concept_atoms(cui)
            
            if not atoms_result:
                return []
            
            synonyms = []
            for atom in atoms_result:
                name = atom.get('name', '')
                if name and name not in synonyms:
                    synonyms.append(name)
            
            return synonyms
            
        except Exception as e:
            logger.error(f"Error getting synonyms for CUI {cui}: {e}")
            return []
    
    def get_concept_relations(self, cui):
        """获取概念关系"""
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return []
            
        try:
            relations_result = self.umls_api.get_concept_relations(cui)
            
            if not relations_result:
                return []
            
            relations = []
            for relation in relations_result:
                rel_type = relation.get('relationLabel', '')
                related_cui = relation.get('relatedId', '')
                related_name = relation.get('relatedIdName', '')
                
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
        """将医学术语标准化为UMLS概念"""
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return entities
            
        normalized_entities = []
        
        for entity in entities:
            try:
                cui = self._get_best_cui_for_term(entity)
                
                if cui:
                    concept_details = self.umls_api.get_concept_details(cui)
                    
                    if concept_details:
                        preferred_name = concept_details.get('name', entity)
                        normalized_entities.append(preferred_name)
                        logger.debug(f"标准化: {entity} -> {preferred_name} (CUI: {cui})")
                    else:
                        normalized_entities.append(entity)
                else:
                    normalized_entities.append(entity)
                    
            except Exception as e:
                logger.error(f"Error normalizing entity '{entity}': {e}")
                normalized_entities.append(entity)
        
        return normalized_entities
    
    def get_semantic_variants(self, entity):
        """获取实体的语义变体"""
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return [entity]
            
        try:
            cui = self._get_best_cui_for_term(entity)
            if not cui:
                return [entity]
            
            synonyms = self.get_concept_synonyms(cui)
            relations = self.get_concept_relations(cui)
            related_terms = []
            
            for relation in relations:
                if relation['relation_type'] in ['SY', 'PT', 'equivalent_to']:
                    related_terms.append(relation['related_name'])
            
            variants = [entity] + synonyms + related_terms
            
            unique_variants = []
            seen = set()
            
            for variant in variants:
                if variant and variant.lower() not in seen and len(variant) > 2:
                    seen.add(variant.lower())
                    unique_variants.append(variant)
            
            return unique_variants[:10]
            
        except Exception as e:
            logger.error(f"Error getting semantic variants for '{entity}': {e}")
            return [entity]
    
    def get_concept_hierarchy(self, entity):
        """获取概念层次结构"""
        if not ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            return {}
            
        try:
            cui = self._get_best_cui_for_term(entity)
            if not cui:
                return {}
            
            relations = self.get_concept_relations(cui)
            hierarchy = {
                'broader': [],
                'narrower': [],
                'related': []
            }
            
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
        """增强的实体链接"""
        if not ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
            return {}
            
        try:
            umls_kg = []
            for entity in entities:
                concepts = self.umls_api.search_concepts(entity)
                if concepts and 'results' in concepts:
                    umls_kg.extend([concept['name'] for concept in concepts['results'][:5]])
            
            linking_results = self.enhanced_entity_linking.multi_strategy_linking(
                entities, context, umls_kg
            )
            
            return linking_results
            
        except Exception as e:
            logger.error(f"Error in enhanced entity linking: {e}")
            return {}
    
    def adaptive_knowledge_selection(self, question_types, entities):
        """自适应知识选择"""
        if not ABLATION_CONFIG['USE_ADAPTIVE_UMLS']:
            return []
            
        try:
            selected_knowledge = []
            
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
        """应用医学推理规则扩展知识"""
        if not ABLATION_CONFIG['USE_REASONING_RULES']:
            logger.info("🔬 Medical reasoning rules disabled in ablation study")
            return knowledge_triples
            
        expanded_triples = knowledge_triples.copy()
        reasoning_log = []
        
        transitive_triples = self._apply_transitivity(knowledge_triples, max_hops)
        expanded_triples.extend(transitive_triples)
        reasoning_log.extend([('transitivity', len(transitive_triples))])
        
        inverse_triples = self._apply_inverse_relations(knowledge_triples)
        expanded_triples.extend(inverse_triples)
        reasoning_log.extend([('inverse', len(inverse_triples))])
        
        semantic_triples = self._apply_semantic_implications(knowledge_triples)
        expanded_triples.extend(semantic_triples)
        reasoning_log.extend([('semantic', len(semantic_triples))])
        
        hierarchical_triples = self._apply_hierarchical_reasoning(knowledge_triples)
        expanded_triples.extend(hierarchical_triples)
        reasoning_log.extend([('hierarchical', len(hierarchical_triples))])
        
        unique_triples = self._deduplicate_triples(expanded_triples)
        
        logger.info(f"推理扩展: {reasoning_log}")
        logger.info(f"原始三元组: {len(knowledge_triples)}, 扩展后: {len(unique_triples)}")
        
        return unique_triples
    
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

# ========================= 初始化增强模块 =========================
umls_api_key = "7cce913d-29bf-459f-aa9a-2ba57d6efccf"
umls_normalizer = UMLSNormalizer(umls_api_key)
medical_reasoning_rules = MedicalReasoningRules(umls_normalizer)
multi_hop_reasoner = MultiHopReasoning(max_hops=3, umls_normalizer=umls_normalizer)


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
    """获取医学术语的同义词"""
    entity_lower = entity.lower()
    synonyms = [entity]
    
    if ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
        try:
            umls_variants = umls_normalizer.get_semantic_variants(entity)
            synonyms.extend(umls_variants)
            logger.debug(f"UMLS variants for '{entity}': {umls_variants}")
        except Exception as e:
            logger.error(f"Error getting UMLS variants for '{entity}': {e}")
    
    for key, synonym_list in MEDICAL_SYNONYMS.items():
        if key in entity_lower or entity_lower in synonym_list:
            synonyms.extend(synonym_list)
    
    if ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
        try:
            normalized_synonyms = umls_normalizer.normalize_medical_terms(synonyms)
            synonyms.extend(normalized_synonyms)
        except Exception as e:
            logger.error(f"Error normalizing synonyms for '{entity}': {e}")
    
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

def chat_35(prompt):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
    {"role": "user", "content": prompt}
    ])
    return completion.choices[0].message.content

def chat_4(prompt):
    completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
    {"role": "user", "content": prompt}
    ])
    return completion.choices[0].message.content

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
    
    for kg_entity in question_kg:
        try:
            if kg_entity in keyword_embeddings["keywords"]:
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
            else:
                best_match_idx = None
                best_similarity = 0
                for idx, keyword in enumerate(keyword_embeddings["keywords"]):
                    if kg_entity.lower() in keyword.lower():
                        similarity = 0.8
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
            
            if similarity_score >= 0.6:
                candidate_entity = entity_embeddings["entities"][best_idx]
                if candidate_entity not in match_kg:
                    match_kg.append(candidate_entity)
                    entity_confidence_scores.append(float(similarity_score))
                    logger.debug(f"Basic matched: {kg_entity} -> {candidate_entity} (score: {similarity_score:.3f})")
                
        except Exception as e:
            logger.error(f"Error in basic entity matching for {kg_entity}: {e}")
            continue
    
    return match_kg, entity_confidence_scores

def enhanced_entity_matching(question_kg, entity_embeddings, keyword_embeddings, question_text=""):
    """增强的实体匹配，集成真实UMLS API和新优化"""
    
    if not any([
        ABLATION_CONFIG['USE_HIERARCHICAL_KG'],
        ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING'],
        ABLATION_CONFIG['USE_ADAPTIVE_UMLS'],
        ABLATION_CONFIG['USE_UMLS_NORMALIZATION'],
        ABLATION_CONFIG['USE_REASONING_RULES']
    ]):
        logger.info("🔬 Using basic entity matching (all enhancements disabled)")
        return basic_entity_matching(question_kg, entity_embeddings, keyword_embeddings, question_text)
    
    match_kg = []
    entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
    entity_confidence_scores = []
    
    question_types = semantic_question_classifier.identify_question_type(question_text)
    
    expanded_entities = []
    for kg_entity in question_kg:
        expanded_entity = expand_medical_abbreviations(kg_entity)
        expanded_entities.append(expanded_entity)
        
        if ABLATION_CONFIG['USE_UMLS_NORMALIZATION']:
            synonyms = get_medical_synonyms(kg_entity)
            expanded_entities.extend(synonyms)
    
    if ABLATION_CONFIG['USE_MULTI_STRATEGY_LINKING']:
        try:
            enhanced_links = umls_normalizer.enhanced_entity_linking_method(
                expanded_entities, question_text, question_types
            )
            
            for entity, link_info in enhanced_links.items():
                if link_info.get('final_score', 0) > 0.6:
                    expanded_entities.append(entity)
                    
        except Exception as e:
            logger.error(f"Error in enhanced entity linking: {e}")
    
    if ABLATION_CONFIG['USE_ADAPTIVE_UMLS']:
        try:
            adaptive_knowledge = umls_normalizer.adaptive_knowledge_selection(
                question_types, expanded_entities
            )
            
            for knowledge_item in adaptive_knowledge:
                if isinstance(knowledge_item, dict):
                    related_name = knowledge_item.get('related_name', '')
                    if related_name:
                        expanded_entities.append(related_name)
                        
        except Exception as e:
            logger.error(f"Error in adaptive knowledge selection: {e}")
    
    if ABLATION_CONFIG['USE_REASONING_RULES']:
        try:
            temp_triples = [[entity, 'mentions', 'question'] for entity in expanded_entities]
            reasoned_triples = medical_reasoning_rules.apply_reasoning_rules(temp_triples)
            
            for triple in reasoned_triples:
                if len(triple) >= 3:
                    expanded_entities.extend([triple[0], triple[2]])
        except Exception as e:
            logger.error(f"Error in reasoning-based entity expansion: {e}")
    
    seen = set()
    unique_entities = []
    for entity in expanded_entities:
        if entity.lower() not in seen:
            seen.add(entity.lower())
            unique_entities.append(entity)
    
    logger.info(f"Original entities: {question_kg}")
    logger.info(f"Expanded entities (with optimizations): {unique_entities[:10]}...")
    
    is_negation = has_negation(question_text)
    if 'exception' in question_types or is_negation:
        similarity_threshold = MIN_SIMILARITY_THRESHOLD * 0.8
    else:
        similarity_threshold = MIN_SIMILARITY_THRESHOLD
    
    for kg_entity in unique_entities:
        try:
            if kg_entity in keyword_embeddings["keywords"]:
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
            else:
                best_match_idx = None
                best_similarity = 0
                for idx, keyword in enumerate(keyword_embeddings["keywords"]):
                    if kg_entity.lower() in keyword.lower() or keyword.lower() in kg_entity.lower():
                        similarity = len(set(kg_entity.lower().split()) & set(keyword.lower().split())) / len(set(kg_entity.lower().split()) | set(keyword.lower().split()))
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_idx = idx
                
                if best_match_idx is None or best_similarity < 0.3:
                    continue
                keyword_index = best_match_idx
            
            kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

            kg_entity_emb_norm = kg_entity_emb / np.linalg.norm(kg_entity_emb)
            entity_embeddings_norm = entity_embeddings_emb.values / np.linalg.norm(entity_embeddings_emb.values, axis=1, keepdims=True)
            
            cos_similarities = np.dot(entity_embeddings_norm, kg_entity_emb_norm)
            
            top_indices = np.argsort(cos_similarities)[::-1]
            
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
    
    if entity_confidence_scores:
        avg_confidence = np.mean(entity_confidence_scores)
        logger.info(f"Entity matching average confidence: {avg_confidence:.3f}")
    
    return match_kg, entity_confidence_scores

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

def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim

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

def autowrap_text(text, font, max_width):
    text_lines = []
    if font.getsize(text)[0] <= max_width:
        text_lines.append(text)
    else:
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + ' '
                i += 1
            if not line:
                line = words[i]
                i += 1
            text_lines.append(line)
    return text_lines

def enhanced_final_answer(question_text, response_of_KG_list_path, response_of_KG_neighbor):
    """增强的最终答案生成，移除置信度计算，直接使用投票机制"""
    if not ABLATION_CONFIG['USE_ENHANCED_ANSWER_GEN']:
        logger.info("🔬 Using basic final answer generation")
        return basic_final_answer(question_text, response_of_KG_list_path, response_of_KG_neighbor)
    
    if response_of_KG_list_path == []:
        response_of_KG_list_path = ''
    if response_of_KG_neighbor == []:
        response_of_KG_neighbor = ''
    
    # 保留：问题类型识别和否定词处理
    question_types = semantic_question_classifier.identify_question_type(question_text)
    has_neg = has_negation(question_text)
    
    # 保留：KG引导推理
    try:
        kg_subgraph = []
        
        if response_of_KG_list_path:
            path_lines = response_of_KG_list_path.split('\n')
            for line in path_lines:
                if '->' in line:
                    parts = line.split('->')
                    if len(parts) >= 3:
                        kg_subgraph.append([parts[0].strip(), parts[1].strip(), parts[2].strip()])
        
        if response_of_KG_neighbor:
            neighbor_lines = response_of_KG_neighbor.split('\n')
            for line in neighbor_lines:
                if '->' in line:
                    parts = line.split('->')
                    if len(parts) >= 3:
                        kg_subgraph.append([parts[0].strip(), parts[1].strip(), parts[2].strip()])
        
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
            final_prompts = [
                "The final answer (output the letter option) is:",
                "Based on the hierarchical analysis above, the correct answer is:",
                "Therefore, using multi-strategy reasoning, the answer choice is:"
            ]
            
            messages = [
                SystemMessage(content="You are an excellent AI assistant specialized in medical question answering with access to UMLS standardized medical knowledge and hierarchical reasoning capabilities"),
                HumanMessage(content=f'Question: {question_text}'),
                AIMessage(content=f"Medical knowledge:\n\n" + 
                         f'###Path-based Evidence: {response_of_KG_list_path}\n\n' + 
                         f'###Neighbor-based Evidence: {response_of_KG_neighbor}'),
                AIMessage(content=f"Analysis: {output_CoT}"),
                AIMessage(content=final_prompts[attempt % len(final_prompts)])
            ]
            
            result = chat(messages)
            if result.content is not None and len(result.content.strip()) > 0:
                answer_match = re.search(r'\b([A-E])\b', result.content)
                if answer_match:
                    answers.append(answer_match.group(1))
                else:
                    answers.append(result.content.strip()[:10])
                    
        except Exception as e:
            logger.error(f"Final answer attempt {attempt + 1} failed: {e}")
            sleep(3)
    
    # 简化：直接投票选择，移除置信度计算
    if answers:
        answer_counts = Counter(answers)
        most_common_answer = answer_counts.most_common(1)[0][0]
        
        logger.info(f"Voting results: {dict(answer_counts)}, Selected: {most_common_answer}")
        return most_common_answer
    
    logger.error("All final answer attempts failed")
    return "A"

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

@retry_on_failure()
def prompt_document(question,instruction):
    """原始文档提示模板"""
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

    prompt = PromptTemplate(
        template = template,
        input_variables = ["question","instruction"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(question = question,
                                 instruction = instruction)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(question = question,\
                                                        instruction = instruction,\
                                                        text={})

    response_document_bm25 = chat(chat_prompt_with_values.to_messages()).content
    return response_document_bm25

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

def get_file_hash(file_path):
    """获取文件哈希值，用于检测文件是否变化"""
    import hashlib
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

if __name__ == "__main__":
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
        use_semantic_matching=True,    # 启用语义匹配
        similarity_threshold=0.7       # 设置相似度阈值
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
        similarity_threshold=0.4
    )

    medical_reasoning_rules.initialize_kg_guided_reasoning(flat_kg_triples, chat)

    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset}")
        processor = dataset2processor[dataset]()
        data = processor.load_dataset()

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

            match_kg, confidence_scores = enhanced_entity_matching(
                question_kg, entity_embeddings, keyword_embeddings, input_text[0])

            if len(match_kg) < 2:
                logger.warning(f"Insufficient entities matched for question: {input_text[0][:100]}...")
                match_kg.extend(question_kg[:2])

            # 4. 增强的neo4j知识图谱路径查找
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
            
            # 5. 增强的neo4j知识图谱邻居实体
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

            # 6. 增强的知识图谱路径基础提示生成
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

            # 8. 增强的基于提示的医学对话答案生成（移除了置信度计算）
            output_all = enhanced_final_answer(input_text[0], response_of_KG_list_path, response_of_KG_neighbor)
            
            # 重试屏蔽位置
            if is_unable_to_answer(output_all):
                logger.warning("Final answer validation failed, retrying...")
                output_all = enhanced_final_answer(input_text[0], response_of_KG_list_path, response_of_KG_neighbor)
            # 结束位置

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