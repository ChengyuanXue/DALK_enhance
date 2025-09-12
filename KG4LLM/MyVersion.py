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

dataset2processor = {
    'medmcqa': medmcqaZeroshotsProcessor,
    'medqa':medqaZeroshotsProcessor,
    'mmlu': mmluZeroshotsProcessor,
    'qa4mre':qa4mreZeroshotsProcessor
}
datasets = ['medqa', 'medmcqa', 'mmlu', 'qa4mre']

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
    # 首先检查是否为None或NaN
    if pd.isna(head) or pd.isna(relation) or pd.isna(tail):
        return False
    
    # 转换为字符串以防万一
    head = str(head).strip() if head is not None else ""
    relation = str(relation).strip() if relation is not None else ""
    tail = str(tail).strip() if tail is not None else ""
    
    # Basic validation rules
    if not head or not relation or not tail:
        return False
    
    # Check for meaningful content (not just single characters or numbers)
    if len(head) < 2 or len(tail) < 2:
        return False
    
    # Check for common noise patterns
    noise_patterns = ['http', 'www', '@', '#', '___', '...', 'nan', 'none']
    for pattern in noise_patterns:
        if pattern in head.lower() or pattern in tail.lower():
            return False
    
    return True

def enhanced_entity_matching(question_kg, entity_embeddings, keyword_embeddings):
    """Enhanced entity matching with improved similarity calculation and validation"""
    match_kg = []
    entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
    entity_confidence_scores = []
    
    for kg_entity in question_kg:
        try:
            keyword_index = keyword_embeddings["keywords"].index(kg_entity)
            kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

            # Enhanced cosine similarity calculation with normalization
            kg_entity_emb_norm = kg_entity_emb / np.linalg.norm(kg_entity_emb)
            entity_embeddings_norm = entity_embeddings_emb.values / np.linalg.norm(entity_embeddings_emb.values, axis=1, keepdims=True)
            
            cos_similarities = np.dot(entity_embeddings_norm, kg_entity_emb_norm)
            
            # Find multiple high-confidence matches
            top_indices = np.argsort(cos_similarities)[::-1]
            
            best_match_found = False
            for idx in top_indices[:5]:  # Check top 5 matches
                similarity_score = cos_similarities[idx]
                candidate_entity = entity_embeddings["entities"][idx]
                
                # Apply confidence threshold and avoid duplicates
                if (similarity_score >= MIN_SIMILARITY_THRESHOLD and 
                    candidate_entity not in match_kg):
                    match_kg.append(candidate_entity)
                    entity_confidence_scores.append(float(similarity_score))  # 转换为Python float
                    best_match_found = True
                    break
            
            if not best_match_found:
                logger.warning(f"No high-confidence match found for entity: {kg_entity}")
                
        except ValueError:
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

def enhanced_find_shortest_path(start_entity_name, end_entity_name, candidate_list):
    """Enhanced path finding with better validation and quality control"""
    global exist_entity
    paths_with_scores = []
    
    with driver.session() as session:
        try:
            result = session.run(
                "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
                "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
                "RETURN p LIMIT 10",  # Limit results for efficiency
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
                        
                        # Calculate path quality based on relation types
                        if any(keyword in relation_type.lower() for keyword in ['cause', 'treat', 'affect', 'associate']):
                            path_quality_score += 1
               
                # Validate path quality
                path_str = ""
                for i in range(len(entities)):
                    entities[i] = entities[i].replace("_"," ")
                    
                    if entities[i] in candidate_list:
                        short_path = 1
                        exist_entity = entities[i]
                        path_quality_score += 2  # Bonus for candidate entities
                        
                    path_str += entities[i]
                    if i < len(relations):
                        relations[i] = relations[i].replace("_"," ")
                        path_str += "->" + relations[i] + "->"
                
                # Store path with quality score
                paths_with_scores.append((path_str, path_quality_score))
                
                if short_path == 1:
                    # Sort by quality and return best paths
                    paths_with_scores.sort(key=lambda x: x[1], reverse=True)
                    paths = [path[0] for path in paths_with_scores[:5]]
                    break
            
            if not paths and paths_with_scores:
                # If no short path found, return highest quality paths
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

def find_shortest_path(start_entity_name, end_entity_name,candidate_list):
    """Original function with enhanced error handling"""
    return enhanced_find_shortest_path(start_entity_name, end_entity_name, candidate_list)

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

def enhanced_get_entity_neighbors(entity_name: str, disease_flag) -> Tuple[List[List[str]], List[str]]:
    """Enhanced neighbor extraction with quality filtering"""
    disease = []
    neighbor_list = []
    
    query = """
    MATCH (e:Entity)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type,
           collect(n.name) AS neighbor_entities
    ORDER BY size(collect(n.name)) DESC
    LIMIT 20
    """
    
    try:
        result = session.run(query, entity_name=entity_name)
        relation_quality_scores = {}
        
        for record in result:
            rel_type = record["relationship_type"]
            
            if disease_flag == 1 and rel_type == 'has_symptom':
                continue

            neighbors = record["neighbor_entities"]
            
            # Calculate relation quality score
            quality_score = 0
            if "disease" in rel_type.replace("_"," ").lower():
                disease.extend(neighbors)
                quality_score += 2
            elif any(keyword in rel_type.lower() for keyword in ['cause', 'treat', 'affect', 'associate']):
                quality_score += 1.5
            else:
                quality_score += 1
                
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

def get_entity_neighbors(entity_name: str,disease_flag) -> List[List[str]]:
    """Original function signature maintained"""
    neighbor_list, disease = enhanced_get_entity_neighbors(entity_name, disease_flag)
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
    """Enhanced reranking with quality assessment"""
    template = """
    There is a question and some knowledge graph. The knowledge graphs follow entity->relationship->entity list format.
    \n\n
    ##Graph: {graph}
    \n\n
    ##Question: {question}
    \n\n
    Please rerank the knowledge graph and output at most 5 important and relevant triples for solving the given question. Focus on triples that directly relate to the question entities and provide meaningful medical relationships. Output the reranked knowledge in the following format:
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

def enhanced_final_answer(str, response_of_KG_list_path, response_of_KG_neighbor, question_text):
    """Enhanced final answer generation with multi-round validation"""
    if response_of_KG_list_path == []:
        response_of_KG_list_path = ''
    if response_of_KG_neighbor == []:
        response_of_KG_neighbor = ''
    
    # First attempt - CoT reasoning
    messages = [
        SystemMessage(content="You are an excellent AI assistant to answering the following question"),
        HumanMessage(content='Question: ' + question_text),
        AIMessage(content="You have some medical knowledge information in the following:\n\n" + 
                 '###' + response_of_KG_list_path + '\n\n' + '###' + response_of_KG_neighbor),
        HumanMessage(content="Answer: Let's think step by step: ")
    ]
    
    output_CoT = ""
    for retry in range(MAX_RETRIES):
        try:
            result_CoT = chat(messages)
            if result_CoT.content is not None and len(result_CoT.content.strip()) > 10:
                output_CoT = result_CoT.content
                break
            else:
                logger.warning(f"CoT generation attempt {retry + 1} returned insufficient content")
                sleep(RETRY_WAIT_TIME)
        except Exception as e:
            logger.error(f"CoT generation attempt {retry + 1} failed: {e}")
            sleep(RETRY_WAIT_TIME)
    
    if not output_CoT:
        logger.warning("CoT generation failed, proceeding without it")
        output_CoT = "Based on the provided knowledge, let me analyze this step by step."
    
    # Second attempt - Final answer with validation
    messages = [
        SystemMessage(content="You are an excellent AI assistant to answering the following question"),
        HumanMessage(content='Question: ' + question_text),
        AIMessage(content="You have some medical knowledge information in the following:\n\n" + 
                 '###' + response_of_KG_list_path + '\n\n' + '###' + response_of_KG_neighbor),
        AIMessage(content="Answer: Let's think step by step: " + output_CoT),
        AIMessage(content="The final answer (output the letter option) is:")
    ]
    
    # Multiple attempts for final answer with consistency check
    consistent_answers = []
    for attempt in range(3):  # Try 3 times for consistency
        try:
            result = chat(messages)
            if result.content is not None and len(result.content.strip()) > 0:
                # Extract letter answer
                answer_match = re.search(r'\b([A-E])\b', result.content)
                if answer_match:
                    consistent_answers.append(answer_match.group(1))
                consistent_answers.append(result.content.strip())
                if len(consistent_answers) >= 2:
                    break
        except Exception as e:
            logger.error(f"Final answer attempt {attempt + 1} failed: {e}")
            sleep(5)
    
    # Return most consistent answer or first valid answer
    if consistent_answers:
        # Count letter answers for consistency
        letter_answers = [ans for ans in consistent_answers if len(ans) == 1 and ans.isalpha()]
        if letter_answers:
            most_common = Counter(letter_answers).most_common(1)[0][0]
            return most_common
        return consistent_answers[0]
    
    logger.error("All final answer attempts failed")
    return ""

def final_answer(str,response_of_KG_list_path,response_of_KG_neighbor):
    """Original function signature maintained"""
    return enhanced_final_answer(str, response_of_KG_list_path, response_of_KG_neighbor, input_text[0])

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
    
    # 读取CSV文件
    df = pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    
    # 删除包含NaN值的行
    df_clean = df.dropna().copy()  # 使用copy()避免SettingWithCopyWarning
    
    # 转换为字符串并去除空白
    df_clean.loc[:, 'head'] = df_clean['head'].astype(str).str.strip()
    df_clean.loc[:, 'relation'] = df_clean['relation'].astype(str).str.strip()
    df_clean.loc[:, 'tail'] = df_clean['tail'].astype(str).str.strip()
    
    # 删除空字符串行
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
    session.run("MATCH (n) DETACH DELETE n")# clean all

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
        
        # Validate triple before insertion
        if not validate_knowledge_triple(head_name, relation_name, tail_name):
            continue

        # 准备批量查询
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
        
        # 当达到批量大小时执行批量插入
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
                # 如果批量失败，尝试逐个插入
                for q, params in zip(batch_queries, batch_params):
                    try:
                        session.run(q, **params)
                        valid_triples += 1
                    except Exception as single_e:
                        logger.warning(f"Failed to insert single triple: {params['head_name']} -> {params['relation_name']} -> {params['tail_name']}, Error: {single_e}")
            
            # 清空批量缓存
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
            # 如果批量失败，尝试逐个插入
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

            # Enhanced entity matching
            match_kg, confidence_scores = enhanced_entity_matching(
                question_kg, entity_embeddings, keyword_embeddings)

            if len(match_kg) < 2:
                logger.warning(f"Insufficient entities matched for question: {input_text[0][:100]}...")
                match_kg.extend(question_kg[:2])  # Fallback to original entities

            # 4. Enhanced neo4j knowledge graph path finding
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
                        
                        paths, exist_entity = find_shortest_path(start_entity, end_entity, candidate_entity)
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
            
            # 5. Enhanced neo4j knowledge graph neighbor entities
            neighbor_list = []
            neighbor_list_disease = []
            
            for match_entity in match_kg:
                disease_flag = 0
                neighbors, disease = get_entity_neighbors(match_entity, disease_flag)
                neighbor_list.extend(neighbors)

                while disease:
                    new_disease = []
                    for disease_tmp in disease:
                        if disease_tmp in match_kg:
                            new_disease.append(disease_tmp)

                    if new_disease:
                        for disease_entity in new_disease:
                            disease_flag = 1
                            neighbors, disease = get_entity_neighbors(disease_entity, disease_flag)
                            neighbor_list_disease.extend(neighbors)
                    else:
                        for disease_entity in disease:
                            disease_flag = 1
                            neighbors, disease = get_entity_neighbors(disease_entity, disease_flag)
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
                    
                    # Validation and retry for path finding
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
            
            # Validation and retry for neighbor processing
            if is_unable_to_answer(response_of_KG_neighbor):
                logger.warning("Neighbor processing response validation failed, retrying...")
                response_of_KG_neighbor = prompt_neighbor(neighbor_input_sampled)

            # 8. Enhanced prompt-based medical dialogue answer generation
            output_all = final_answer(input_text[0], response_of_KG_list_path, response_of_KG_neighbor)
            
            # Additional validation for final answer
            if is_unable_to_answer(output_all):
                logger.warning("Final answer validation failed, retrying...")
                output_all = final_answer(input_text[0], response_of_KG_list_path, response_of_KG_neighbor)

            ret_parsed, acc_item = processor.parse(output_all, item)
            ret_parsed['path'] = path_sampled if 'path_sampled' in locals() else ""
            ret_parsed['neighbor_input'] = neighbor_input_sampled if 'neighbor_input_sampled' in locals() else ""
            ret_parsed['response_of_KG_list_path'] = response_of_KG_list_path
            ret_parsed['response_of_KG_neighbor'] = response_of_KG_neighbor
            ret_parsed['entity_confidence_scores'] = confidence_scores if 'confidence_scores' in locals() else []
            
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
        
        with open(os.path.join('./Alzheimers/result_chatgpt_mindmap', f"{dataset}_enhanced_reranking.json"), 'w') as f:
            json.dump(generated_data, fp=f)
            
        logger.info(f"Results saved for dataset: {dataset}")

    logger.info("All datasets processed successfully!")
    
    # Close database connection
    driver.close()