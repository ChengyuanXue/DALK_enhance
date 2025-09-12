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
from collections import deque, Counter
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
from sklearn.metrics.pairwise import cosine_similarity
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

from dataset_utils import *
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced API retry configuration
MAX_RETRIES = 60
RETRY_WAIT_TIME = 60
ENTITY_CONFIDENCE_THRESHOLD = 0.85  # Entity matching confidence threshold
KNOWLEDGE_QUALITY_THRESHOLD = 0.7   # Knowledge quality threshold
MIN_SIMILARITY_THRESHOLD = 0.6      # Minimum similarity for entity matching

# ========================= 修改后的UMLS API集成（基于第二个代码的成功方式） =========================

class UMLS_API:
    def __init__(self, api_key, version="current"):
        """初始化UMLS API客户端 - 使用直接API key认证"""
        self.api_key = api_key
        self.version = version
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        
        # 设置session
        self.session = requests.Session()
        self.session.timeout = 120
        
        # 缓存机制
        self.cache = {}
        self.cache_size = 10000
        
        # 测试连接
        try:
            self._test_connection()
            logger.info("UMLS API connection successful")
        except Exception as e:
            logger.warning(f"UMLS API connection failed: {e}. Operating in offline mode.")
    
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
            
            # 缓存结果
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching concepts for '{search_string}': {e}")
            return None
    
    def get_concept_details(self, cui):
        """获取概念详细信息"""
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
        cache_key = f"relations_{cui}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        all_relations = []
        
        try:
            for page in range(1, 6):  # 限制为5页
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
            
        except Exception as e:
            logger.error(f"Error getting relations for CUI {cui}: {e}")
            return []
    
    def get_semantic_types(self, cui):
        """获取语义类型"""
        cache_key = f"semantic_{cui}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            url = f"{self.content_url}/CUI/{cui}/definitions"
            params = {"apiKey": self.api_key}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            response.encoding = "utf-8"
            
            data = response.json()
            result = data.get("result", [])
            
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting semantic types for CUI {cui}: {e}")
            return []

class UMLSNormalizer:
    def __init__(self, api_key):
        """初始化UMLS标准化器"""
        self.umls_api = UMLS_API(api_key)
        self.local_cache = {}
        self.semantic_type_cache = {}
        
        # 医学语义类型优先级
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
        if term in self.local_cache:
            return self.local_cache[term]
        
        try:
            # 搜索概念
            search_results = self.umls_api.search_concepts(term)
            
            if not search_results or 'results' not in search_results:
                return None
            
            # 提取搜索结果
            results = search_results['results']
            if not results:
                return None
            
            # 选择最佳匹配
            best_cui = None
            best_score = 0
            
            for result in results:
                cui = result['ui']
                name = result['name']
                
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
        """计算匹配分数"""
        score = 0
        
        # 精确匹配奖励
        if original_term.lower() == concept_name.lower():
            score += 100
        
        # 部分匹配奖励
        elif original_term.lower() in concept_name.lower():
            score += 50
        elif concept_name.lower() in original_term.lower():
            score += 30
        
        # 单词重叠奖励
        original_words = set(original_term.lower().split())
        concept_words = set(concept_name.lower().split())
        overlap = len(original_words & concept_words)
        score += overlap * 10
        
        # 根部分匹配奖励（去除后缀s, ing等）
        if self._has_root_match(original_term, concept_name):
            score += 20
        
        return score
    
    def _has_root_match(self, term1, term2):
        """检查词根匹配"""
        # 简单的词根匹配逻辑
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
        normalized_entities = []
        
        for entity in entities:
            try:
                # 获取最佳CUI
                cui = self._get_best_cui_for_term(entity)
                
                if cui:
                    # 获取概念详细信息
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
        try:
            cui = self._get_best_cui_for_term(entity)
            if not cui:
                return [entity]
            
            # 获取同义词
            synonyms = self.get_concept_synonyms(cui)
            
            # 获取相关概念
            relations = self.get_concept_relations(cui)
            related_terms = []
            
            for relation in relations:
                if relation['relation_type'] in ['SY', 'PT', 'equivalent_to']:  # 同义关系
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
            
            return unique_variants[:10]  # 限制返回数量
            
        except Exception as e:
            logger.error(f"Error getting semantic variants for '{entity}': {e}")
            return [entity]
    
    def get_concept_hierarchy(self, entity):
        """获取概念层次结构"""
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

# ========================= 医学推理规则模块（完整版本） =========================

class MedicalReasoningRules:
    def __init__(self, umls_normalizer=None):
        """初始化医学推理规则"""
        self.umls_normalizer = umls_normalizer
        
        # 扩展的推理规则
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
                # 如果A是B的症状，那么B有症状A
                'symptom_of': 'has_symptom',
                'risk_factor_for': 'has_risk_factor',
                'complication_of': 'has_complication'
            },
            'medical_hierarchies': {
                # 疾病层次推理
                'disease_subtype': 'is_type_of',
                'anatomical_part': 'part_of_anatomy',
                'drug_class': 'belongs_to_class'
            }
        }
        
        # 推理置信度权重
        self.confidence_weights = {
            'direct': 1.0,
            'transitive_1hop': 0.8,
            'transitive_2hop': 0.6,
            'inverse': 0.9,
            'semantic': 0.7,
            'hierarchical': 0.75
        }
    
    def apply_reasoning_rules(self, knowledge_triples, max_hops=2):
        """应用医学推理规则扩展知识"""
        expanded_triples = knowledge_triples.copy()
        reasoning_log = []
        
        # 传递性推理
        transitive_triples = self._apply_transitivity(knowledge_triples, max_hops)
        expanded_triples.extend(transitive_triples)
        reasoning_log.extend([('transitivity', len(transitive_triples))])
        
        # 逆关系推理
        inverse_triples = self._apply_inverse_relations(knowledge_triples)
        expanded_triples.extend(inverse_triples)
        reasoning_log.extend([('inverse', len(inverse_triples))])
        
        # 语义蕴涵推理
        semantic_triples = self._apply_semantic_implications(knowledge_triples)
        expanded_triples.extend(semantic_triples)
        reasoning_log.extend([('semantic', len(semantic_triples))])
        
        # 层次推理
        hierarchical_triples = self._apply_hierarchical_reasoning(knowledge_triples)
        expanded_triples.extend(hierarchical_triples)
        reasoning_log.extend([('hierarchical', len(hierarchical_triples))])
        
        # 去重
        unique_triples = self._deduplicate_triples(expanded_triples)
        
        logger.info(f"推理扩展: {reasoning_log}")
        logger.info(f"原始三元组: {len(knowledge_triples)}, 扩展后: {len(unique_triples)}")
        
        return unique_triples
    
    def _apply_transitivity(self, triples, max_hops):
        """应用传递性推理"""
        transitive_triples = []
        
        # 构建关系图
        relation_graph = {}
        for triple in triples:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                if head not in relation_graph:
                    relation_graph[head] = []
                relation_graph[head].append((relation, tail))
        
        # 执行传递性推理
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
                        # 生成传递性三元组
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
        """应用逆关系推理"""
        inverse_triples = []
        
        for triple in triples:
            if len(triple) >= 3:
                head, relation, tail = triple[0], triple[1], triple[2]
                
                # 查找逆关系
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
                
                # 检查语义蕴涵
                for source_rel, target_rel in self.rules['semantic_implications'].items():
                    if source_rel in relation.lower():
                        semantic_triples.append([tail, target_rel, head])
        
        return semantic_triples
    
    def _apply_hierarchical_reasoning(self, triples):
        """应用层次推理"""
        hierarchical_triples = []
        
        if not self.umls_normalizer:
            return hierarchical_triples
        
        # 获取概念层次信息
        entities = set()
        for triple in triples:
            if len(triple) >= 3:
                entities.add(triple[0])
                entities.add(triple[2])
        
        # 为每个实体获取层次信息
        for entity in entities:
            try:
                hierarchy = self.umls_normalizer.get_concept_hierarchy(entity)
                
                # 添加层次关系
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

# ========================= 多跳推理模块（完整版本） =========================

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
    
    def perform_multi_hop_reasoning(self, question, kg_subgraph):
        """执行多跳推理"""
        # 识别问题中的实体
        question_entities = self._extract_question_entities(question)
        
        # 标准化实体（如果有UMLS）
        if self.umls_normalizer:
            normalized_entities = self.umls_normalizer.normalize_medical_terms(question_entities)
            question_entities.extend(normalized_entities)
        
        # 构建推理链
        reasoning_chains = []
        for entity in question_entities:
            chain = self._build_reasoning_chain(entity, kg_subgraph, self.max_hops)
            if chain:
                reasoning_chains.append(chain)
        
        # 融合推理结果
        final_answer = self._fuse_reasoning_chains(reasoning_chains, question)
        return final_answer
    
    def _extract_question_entities(self, question):
        """从问题中提取实体"""
        # 简化的实体提取（实际应用中应使用NER）
        entities = []
        
        # 使用正则表达式和已知医学术语
        medical_terms = [
            'alzheimer', 'dementia', 'brain', 'memory', 'cognitive',
            'treatment', 'medication', 'symptom', 'diagnosis', 'disease',
            'protein', 'amyloid', 'tau', 'hippocampus', 'cortex'
        ]
        
        question_lower = question.lower()
        for term in medical_terms:
            if term in question_lower:
                entities.append(term)
        
        # 提取大写开头的词（可能是专有名词）
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
        
        # 构建图结构
        graph = self._build_graph_from_subgraph(kg_subgraph)
        
        # 执行多跳搜索
        for hop in range(1, max_hops + 1):
            hop_paths = self._find_paths_at_hop(graph, start_entity, hop)
            chain['paths'].extend(hop_paths)
        
        # 计算链的置信度
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
        # 重用原有的关系权重计算
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
        
        # 按置信度排序
        reasoning_chains.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 构建最终答案
        answer_components = []
        total_confidence = 0.0
        
        for chain in reasoning_chains[:3]:  # 取前3个最佳链
            if chain['confidence'] > 0.1:  # 置信度阈值
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
        
        # 选择最佳路径
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
        return score / len(path)  # 标准化by路径长度

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

# 医学同义词映射
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

# 否定词列表
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

# ========================= 核心功能函数 =========================

def expand_medical_abbreviations(text):
    """扩展医学缩写词"""
    expanded_text = text
    for abbr, full_form in MEDICAL_ABBREVIATIONS.items():
        pattern = r'\b' + re.escape(abbr) + r'\b'
        expanded_text = re.sub(pattern, full_form, expanded_text, flags=re.IGNORECASE)
    return expanded_text

def get_medical_synonyms(entity):
    """获取医学术语的同义词（集成真实UMLS API）"""
    entity_lower = entity.lower()
    synonyms = [entity]
    
    # 首先使用UMLS获取语义变体
    try:
        umls_variants = umls_normalizer.get_semantic_variants(entity)
        synonyms.extend(umls_variants)
        logger.debug(f"UMLS variants for '{entity}': {umls_variants}")
    except Exception as e:
        logger.error(f"Error getting UMLS variants for '{entity}': {e}")
    
    # 然后使用本地同义词字典
    for key, synonym_list in MEDICAL_SYNONYMS.items():
        if key in entity_lower or entity_lower in synonym_list:
            synonyms.extend(synonym_list)
    
    # 标准化所有同义词
    try:
        normalized_synonyms = umls_normalizer.normalize_medical_terms(synonyms)
        synonyms.extend(normalized_synonyms)
    except Exception as e:
        logger.error(f"Error normalizing synonyms for '{entity}': {e}")
    
    return list(set(synonyms))

def identify_question_type(question):
    """识别问题类型"""
    question_lower = question.lower()
    question_types = []
    
    for q_type, keywords in QUESTION_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in question_lower:
                question_types.append(q_type)
                break
    
    return question_types if question_types else ['general']

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
    """Decorator for adding retry mechanism to functions"""
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

def prompt_extract_keyword(input_text):
    """Original prompt template - unchanged"""
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

    prompt = PromptTemplate(
        template = template,
        input_variables = ["input"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(input = input_text)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(input = input_text,\
                                                        text={})

    response_of_KG = chat(chat_prompt_with_values.to_messages()).content

    question_kg = re.findall(re1,response_of_KG)
    return question_kg

def validate_knowledge_triple(head, relation, tail):
    """Validate the quality of a knowledge triple with NaN handling"""
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

def enhanced_entity_matching(question_kg, entity_embeddings, keyword_embeddings, question_text=""):
    """Enhanced entity matching with real UMLS API integration"""
    match_kg = []
    entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
    entity_confidence_scores = []
    
    # 预处理：扩展缩写词、获取同义词和UMLS标准化
    expanded_entities = []
    for kg_entity in question_kg:
        # 扩展缩写词
        expanded_entity = expand_medical_abbreviations(kg_entity)
        expanded_entities.append(expanded_entity)
        
        # 获取同义词（包含真实UMLS API调用）
        synonyms = get_medical_synonyms(kg_entity)
        expanded_entities.extend(synonyms)
    
    # 应用推理规则扩展实体关系
    try:
        # 构建临时三元组用于推理
        temp_triples = [[entity, 'mentions', 'question'] for entity in expanded_entities]
        reasoned_triples = medical_reasoning_rules.apply_reasoning_rules(temp_triples)
        
        # 从推理结果中提取新实体
        for triple in reasoned_triples:
            if len(triple) >= 3:
                expanded_entities.extend([triple[0], triple[2]])
    except Exception as e:
        logger.error(f"Error in reasoning-based entity expansion: {e}")
    
    # 去重并保持原始顺序
    seen = set()
    unique_entities = []
    for entity in expanded_entities:
        if entity.lower() not in seen:
            seen.add(entity.lower())
            unique_entities.append(entity)
    
    logger.info(f"Original entities: {question_kg}")
    logger.info(f"Expanded entities (with real UMLS): {unique_entities[:10]}...")  # 只显示前10个
    
    # 识别问题类型以调整匹配策略
    question_types = identify_question_type(question_text)
    is_negation = has_negation(question_text)
    
    # 动态调整相似度阈值
    if 'exception' in question_types or is_negation:
        similarity_threshold = MIN_SIMILARITY_THRESHOLD * 0.8
    else:
        similarity_threshold = MIN_SIMILARITY_THRESHOLD
    
    for kg_entity in unique_entities:
        try:
            if kg_entity in keyword_embeddings["keywords"]:
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
            else:
                # 模糊匹配
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

            # Enhanced cosine similarity calculation with normalization
            kg_entity_emb_norm = kg_entity_emb / np.linalg.norm(kg_entity_emb)
            entity_embeddings_norm = entity_embeddings_emb.values / np.linalg.norm(entity_embeddings_emb.values, axis=1, keepdims=True)
            
            cos_similarities = np.dot(entity_embeddings_norm, kg_entity_emb_norm)
            
            # Find multiple high-confidence matches
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
    
    # Log matching quality
    if entity_confidence_scores:
        avg_confidence = np.mean(entity_confidence_scores)
        logger.info(f"Entity matching average confidence: {avg_confidence:.3f}")
    
    return match_kg, entity_confidence_scores

def enhanced_find_shortest_path(start_entity_name, end_entity_name, candidate_list, question_types=[]):
    """Enhanced path finding with medical knowledge weighting"""
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
                        
                        # 增强的路径质量评分
                        relation_weight = calculate_relation_weight(relation_type)
                        path_quality_score += relation_weight
                        
                        # 根据问题类型调整权重
                        if question_types:
                            if 'treatment' in question_types and 'treat' in relation_type.lower():
                                path_quality_score += 1.0
                            elif 'causation' in question_types and 'cause' in relation_type.lower():
                                path_quality_score += 1.0
                            elif 'symptom' in question_types and 'symptom' in relation_type.lower():
                                path_quality_score += 1.0
               
                # Validate path quality
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
                length_penalty = path_length * 0.1
                final_score = path_quality_score - length_penalty
                
                paths_with_scores.append((path_str, final_score))
                
                if short_path == 1:
                    paths_with_scores.sort(key=lambda x: x[1], reverse=True)
                    paths = [path[0] for path in paths_with_scores[:5]]
                    break
            
            if not paths and paths_with_scores:
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
    """Original function with enhanced implementation"""
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
    """Enhanced neighbor extraction with question-type aware filtering"""
    disease = []
    neighbor_list = []
    
    # 根据问题类型调整查询限制
    limit = 25 if any(q_type in ['treatment', 'causation'] for q_type in question_types) else 20
    
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
            
            # 增强的关系质量评分
            quality_score = calculate_relation_weight(rel_type)
            
            # 根据问题类型调整评分
            if question_types:
                if 'treatment' in question_types and 'treat' in rel_type.lower():
                    quality_score += 1.0
                elif 'causation' in question_types and 'cause' in rel_type.lower():
                    quality_score += 1.0
                elif 'symptom' in question_types and 'symptom' in rel_type.lower():
                    quality_score += 1.0
            
            if "disease" in rel_type.replace("_"," ").lower():
                disease.extend(neighbors)
                quality_score += 1.0
                
            # Filter out low-quality neighbors
            filtered_neighbors = []
            for neighbor in neighbors:
                if validate_knowledge_triple(entity_name, rel_type, neighbor):
                    filtered_neighbors.append(neighbor)
            
            if filtered_neighbors:
                neighbor_entry = [entity_name.replace("_"," "), rel_type.replace("_"," "), 
                                ','.join([x.replace("_"," ") for x in filtered_neighbors])]
                neighbor_list.append(neighbor_entry)
                relation_quality_scores[len(neighbor_list)-1] = quality_score
        
        # Sort neighbors by quality score
        if relation_quality_scores:
            sorted_indices = sorted(relation_quality_scores.keys(), 
                                  key=lambda k: relation_quality_scores[k], reverse=True)
            neighbor_list = [neighbor_list[i] for i in sorted_indices]
    
    except Exception as e:
        logger.error(f"Error getting entity neighbors: {e}")
    
    return neighbor_list, disease

def get_entity_neighbors(entity_name: str, disease_flag, question_types=[]) -> List[List[str]]:
    """Original function signature maintained"""
    neighbor_list, disease = enhanced_get_entity_neighbors(entity_name, disease_flag, question_types)
    return neighbor_list, disease

@retry_on_failure()
def prompt_path_finding(path_input):
    """Original prompt template with retry mechanism"""
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
    """Original prompt template with retry mechanism"""
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
    """Original prompt template with retry mechanism"""
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
    """Enhanced reranking with medical knowledge awareness and multi-hop reasoning"""
    # 识别问题类型和否定词
    question_types = identify_question_type(question)
    has_neg = has_negation(question)
    
    # 应用多跳推理增强知识选择
    try:
        # 解析图结构用于多跳推理
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
    
    # 根据问题类型调整提示模板
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

    # Multiple attempts with validation
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
    """Original function with enhanced implementation"""
    return enhanced_self_knowledge_retrieval_reranking(graph, question)

def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim

def enhanced_is_unable_to_answer(response):
    """Enhanced validation of response quality"""
    if not response or len(response.strip()) < 5:
        return True
    
    # Check for common failure patterns
    failure_patterns = [
        "i don't know", "cannot answer", "insufficient information",
        "unable to determine", "not enough context", "cannot provide"
    ]
    
    response_lower = response.lower()
    for pattern in failure_patterns:
        if pattern in response_lower:
            return True
    
    # Original logic for backward compatibility
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
    """Original function with enhanced implementation"""
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

def calculate_answer_confidence(knowledge_coverage, path_quality, consistency_score):
    """计算答案置信度"""
    confidence = (0.4 * knowledge_coverage + 
                 0.35 * path_quality + 
                 0.25 * consistency_score)
    return min(confidence, 1.0)

def enhanced_final_answer(question_text, response_of_KG_list_path, response_of_KG_neighbor):
    """Enhanced final answer generation with multiple validation strategies"""
    if response_of_KG_list_path == []:
        response_of_KG_list_path = ''
    if response_of_KG_neighbor == []:
        response_of_KG_neighbor = ''
    
    # 识别问题类型
    question_types = identify_question_type(question_text)
    has_neg = has_negation(question_text)
    
    # 根据问题类型调整推理策略
    if has_neg or 'exception' in question_types:
        reasoning_instruction = "Pay special attention to negation words and identify what should be EXCLUDED or what is NOT associated with the topic."
    else:
        reasoning_instruction = "Focus on positive associations and direct relationships."
    
    # First attempt - Enhanced CoT reasoning
    messages = [
        SystemMessage(content="You are an excellent AI assistant specialized in medical question answering with access to UMLS standardized medical knowledge"),
        HumanMessage(content=f'Question: {question_text}'),
        AIMessage(content=f"You have some medical knowledge information in the following:\n\n" + 
                 f'###Path-based Evidence: {response_of_KG_list_path}\n\n' + 
                 f'###Neighbor-based Evidence: {response_of_KG_neighbor}'),
        HumanMessage(content=f"Answer: Let's think step by step. {reasoning_instruction} ")
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
    
    # Multiple answer generation for consistency check
    answers = []
    for attempt in range(3):
        try:
            final_prompts = [
                "The final answer (output the letter option) is:",
                "Based on the analysis above, the correct answer is:",
                "Therefore, the answer choice is:"
            ]
            
            messages = [
                SystemMessage(content="You are an excellent AI assistant specialized in medical question answering with access to UMLS standardized medical knowledge"),
                HumanMessage(content=f'Question: {question_text}'),
                AIMessage(content=f"Medical knowledge:\n\n" + 
                         f'###Path-based Evidence: {response_of_KG_list_path}\n\n' + 
                         f'###Neighbor-based Evidence: {response_of_KG_neighbor}'),
                AIMessage(content=f"Analysis: {output_CoT}"),
                AIMessage(content=final_prompts[attempt % len(final_prompts)])
            ]
            
            result = chat(messages)
            if result.content is not None and len(result.content.strip()) > 0:
                # Extract letter answer
                answer_match = re.search(r'\b([A-E])\b', result.content)
                if answer_match:
                    answers.append(answer_match.group(1))
                else:
                    answers.append(result.content.strip()[:10])
                    
        except Exception as e:
            logger.error(f"Final answer attempt {attempt + 1} failed: {e}")
            sleep(3)
    
    # 一致性分析和最终决策
    if answers:
        answer_counts = Counter(answers)
        most_common_answer, most_common_count = answer_counts.most_common(1)[0]
        
        consistency_score = most_common_count / len(answers)
        knowledge_coverage = min(len(response_of_KG_list_path) / 200, 1.0) * 0.5 + min(len(response_of_KG_neighbor) / 200, 1.0) * 0.5
        
        path_quality = 0.5
        if response_of_KG_list_path and response_of_KG_neighbor:
            path_quality = 0.8
        elif response_of_KG_list_path or response_of_KG_neighbor:
            path_quality = 0.6
        
        confidence = calculate_answer_confidence(knowledge_coverage, path_quality, consistency_score)
        
        logger.info(f"Answer confidence: {confidence:.3f}, consistency: {consistency_score:.3f}")
        
        if confidence > 0.6 or consistency_score >= 0.6:
            return most_common_answer
        else:
            valid_answers = [ans for ans in answers if len(ans) == 1 and ans.isalpha()]
            if valid_answers:
                return min(valid_answers)
            return most_common_answer
    
    logger.error("All final answer attempts failed")
    return "A"

def final_answer(str, response_of_KG_list_path, response_of_KG_neighbor):
    """Original function signature maintained"""
    return enhanced_final_answer(input_text[0], response_of_KG_list_path, response_of_KG_neighbor)

@retry_on_failure()
def prompt_document(question,instruction):
    """Original prompt template with retry mechanism"""
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
    """Load and clean knowledge graph triples from CSV file"""
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

if __name__ == "__main__":
    # Configure third-party API
    openai.api_key = "sk-P4hNAfoKF4JLckjCuE99XbaN4bZIORZDPllgpwh6PnYWv4cj"
    openai.api_base = "https://aiyjg.lol/v1"
    
    # Set environment variables
    os.environ['OPENAI_API_KEY'] = openai.api_key

    # 1. Build neo4j knowledge graph datasets
    uri = "bolt://localhost:7688"
    username = "neo4j"
    password = "Cyber@511"

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()

    ##############################build KG 
    logger.info("Cleaning existing knowledge graph...")
    session.run("MATCH (n) DETACH DELETE n")

    # 使用新的函数加载和清理数据
    df_clean = load_and_clean_triples('./Alzheimers/train_s2s.txt')

    # 批量插入优化
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
    
    # 处理剩余的查询
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

    # 2. OpenAI API based keyword extraction and match entities
    OPENAI_API_KEY = openai.api_key
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo', temperature=0.7)

    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"

    logger.info("Loading embeddings...")
    with open('./Alzheimers/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)
        
    with open('./Alzheimers/keyword_embeddings.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)

    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset}")
        processor = dataset2processor[dataset]()
        data = processor.load_dataset()

        acc, total_num = 0, 0
        generated_data = []

        for item in tqdm(data, desc=f"Processing {dataset}"):
            input_text = [processor.generate_prompt(item)]
            entity_list = item['entity'].split('\n')
            question_kg = []
            
            for entity in entity_list:
                try:
                    entity = entity.split('.')[1].strip()
                    question_kg.append(entity)
                except:
                    continue

            # 识别问题类型
            question_types = identify_question_type(input_text[0])
            logger.info(f"Question types identified: {question_types}")

            # Enhanced entity matching with real UMLS API
            match_kg, confidence_scores = enhanced_entity_matching(
                question_kg, entity_embeddings, keyword_embeddings, input_text[0])

            if len(match_kg) < 2:
                logger.warning(f"Insufficient entities matched for question: {input_text[0][:100]}...")
                match_kg.extend(question_kg[:2])

            # 4. Enhanced neo4j knowledge graph path finding with question types
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
                    
                # Enhanced path selection logic
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
            
            # 5. Enhanced neo4j knowledge graph neighbor entities with question types
            neighbor_list = []
            neighbor_list_disease = []
            
            for match_entity in match_kg:
                disease_flag = 0
                neighbors, disease = get_entity_neighbors(match_entity, disease_flag, question_types)
                neighbor_list.extend(neighbors)

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

            # 6. Enhanced knowledge graph path based prompt generation
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
                    
                    if is_unable_to_answer(response_of_KG_list_path):
                        logger.warning("Path finding response validation failed, retrying...")
                        response_of_KG_list_path = prompt_path_finding(path_sampled)
            else:
                response_of_KG_list_path = '{}'

            try:
                response_single_path = prompt_path_finding(single_path)
                if is_unable_to_answer(response_single_path):
                    response_single_path = prompt_path_finding(single_path)
            except:
                response_single_path = ""

            # 7. Enhanced knowledge graph neighbor entities based prompt generation   
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
            
            if is_unable_to_answer(response_of_KG_neighbor):
                logger.warning("Neighbor processing response validation failed, retrying...")
                response_of_KG_neighbor = prompt_neighbor(neighbor_input_sampled)

            # 8. Enhanced prompt-based medical dialogue answer generation
            output_all = enhanced_final_answer(input_text[0], response_of_KG_list_path, response_of_KG_neighbor)
            
            if is_unable_to_answer(output_all):
                logger.warning("Final answer validation failed, retrying...")
                output_all = enhanced_final_answer(input_text[0], response_of_KG_list_path, response_of_KG_neighbor)

            ret_parsed, acc_item = processor.parse(output_all, item)
            ret_parsed['path'] = path_sampled if 'path_sampled' in locals() else ""
            ret_parsed['neighbor_input'] = neighbor_input_sampled if 'neighbor_input_sampled' in locals() else ""
            ret_parsed['response_of_KG_list_path'] = response_of_KG_list_path
            ret_parsed['response_of_KG_neighbor'] = response_of_KG_neighbor
            ret_parsed['entity_confidence_scores'] = confidence_scores if 'confidence_scores' in locals() else []
            ret_parsed['question_types'] = question_types
            
            # 增加真实UMLS标准化结果
            try:
                ret_parsed['umls_normalized_entities'] = umls_normalizer.normalize_medical_terms(question_kg)
                ret_parsed['umls_semantic_variants'] = [umls_normalizer.get_semantic_variants(entity)[:3] for entity in question_kg[:3]]
            except Exception as e:
                logger.error(f"Error in UMLS processing: {e}")
                ret_parsed['umls_normalized_entities'] = question_kg
                ret_parsed['umls_semantic_variants'] = []
            
            # 转换NumPy类型为Python原生类型，确保JSON可序列化
            ret_parsed = convert_numpy_types(ret_parsed)
            
            if ret_parsed['prediction'] in processor.num2answer.values():
                acc += acc_item
                total_num += 1
            generated_data.append(ret_parsed)

        logger.info(f"Dataset: {dataset}")
        logger.info(f"Accuracy: {acc/total_num:.4f} ({acc}/{total_num})")

        # Ensure output directory exists
        os.makedirs('./Alzheimers/result_chatgpt_mindmap', exist_ok=True)
        
        with open(os.path.join('./Alzheimers/result_chatgpt_mindmap', f"{dataset}_core_umls_enhanced.json"), 'w') as f:
            json.dump(generated_data, fp=f)
            
        logger.info(f"Results saved for dataset: {dataset}")

    logger.info("All datasets processed successfully with core UMLS functionality!")
    
    # Close database connection
    driver.close()