# ====================================================================
# DALK项目 - 实体和关键词的语义嵌入向量生成脚本
# 目的：为知识图谱实体和问题关键词生成语义向量表示
# 输入：三元组文件 + NER结果文件
# 输出：实体嵌入向量文件 + 关键词嵌入向量文件
# 应用：用于实体链接、语义相似度计算、知识检索等任务
# ====================================================================

# 导入必要的库
import os                    # 文件系统操作
import json                  # JSON文件处理
import pandas as pd          # 数据处理和CSV/TSV读取
from tqdm import tqdm        # 进度条显示

# 设置CUDA设备，指定使用第0号GPU
# 如果系统有多个GPU，这里指定使用第一个GPU进行计算
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ====================================================================
# 第一步：从知识图谱三元组文件中提取所有实体
# ====================================================================

# 读取DALK生成的训练集三元组文件
# 文件格式：头实体\t关系\t尾实体（制表符分隔）
# header=None表示文件没有列名行
# names参数为三列指定列名
df = pd.read_csv('train_s2s.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])

# 创建集合来存储所有唯一的实体
# 使用set自动去重，避免重复实体
entity = set()

# 遍历DataFrame的每一行，提取头实体和尾实体
for _, item in tqdm(df.iterrows()):
    # item[0]是头实体，item[2]是尾实体
    # 注意：item[1]是关系，这里不需要为关系生成嵌入
    entity.add(item[0])  # 添加头实体
    entity.add(item[2])  # 添加尾实体

# 将集合转换为列表，便于后续批量处理
# 示例实体可能包括：["Alzheimer's disease", "Thiamine", "BACE1", ...]
entity = list(entity)

# ====================================================================
# 第二步：从命名实体识别(NER)结果文件中提取关键词
# ====================================================================

# 创建集合存储所有关键词
keyword = set()

# 遍历result_ner目录中的所有NER结果文件
# result_ner目录包含对问答数据集进行实体识别的结果
for file in os.listdir('result_ner'):
    # 读取每个JSON格式的NER结果文件
    dataset = json.load(open(os.path.join('result_ner', file)))
    
    # 遍历该文件中的每个数据项
    for item in dataset:
        # 每个item包含问题文本和识别出的实体
        # item['entity']包含识别出的实体，可能格式如：
        # "1. Alzheimer's disease\n2. memory\n3. cognitive decline"
        
        # 按换行符分割实体列表
        k_list = item['entity'].split('\n')
        
        # 处理每个实体行
        for k in k_list:
            try:
                # 实体格式可能是："1. Alzheimer's disease"
                # 通过split('.')提取实体名称，去掉编号前缀
                k = k.split('.')[1].strip()  # 提取"."后的部分并去除空格
                keyword.add(k)               # 添加到关键词集合
            except:
                # 如果格式不符合预期（如没有"."），打印出来以便调试
                print(k)
                continue

# 将关键词集合转换为列表
# 示例关键词可能包括：["Alzheimer's disease", "memory", "cognitive decline", ...]
keyword = list(keyword)

# ====================================================================
# 第三步：加载预训练的句子嵌入模型
# ====================================================================

# 导入sentence-transformers库，用于生成高质量的语义嵌入
from sentence_transformers import SentenceTransformer

# 加载预训练的多语言句子嵌入模型
# all-mpnet-base-v2是一个性能优秀的句子嵌入模型，支持多种语言
# 该模型能够将文本转换为768维的密集向量表示
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# 将模型移动到GPU上加速计算
model.to("cuda")

# ====================================================================
# 第四步：生成实体的嵌入向量
# ====================================================================

# 对所有实体进行批量编码，生成嵌入向量
# 参数说明：
# - batch_size=1024: 批处理大小，一次处理1024个实体
# - show_progress_bar=True: 显示处理进度
# - normalize_embeddings=True: 对嵌入向量进行L2归一化，便于计算余弦相似度
embeddings = model.encode(entity, 
                         batch_size=1024, 
                         show_progress_bar=True, 
                         normalize_embeddings=True)

# 创建实体嵌入字典，包含实体列表和对应的嵌入向量
entity_emb_dict = {
    "entities": entity,      # 实体名称列表
    "embeddings": embeddings  # 对应的嵌入向量数组，形状为(N, 768)
}

# 使用pickle序列化并保存实体嵌入数据
# pickle是Python的标准序列化格式，可以高效保存复杂数据结构
import pickle
with open("entity_embeddings.pkl", "wb") as f:
    pickle.dump(entity_emb_dict, f)

# ====================================================================
# 第五步：生成关键词的嵌入向量
# ====================================================================

# 对所有关键词进行批量编码，参数设置与实体编码相同
embeddings = model.encode(keyword, 
                         batch_size=1024, 
                         show_progress_bar=True, 
                         normalize_embeddings=True)

# 创建关键词嵌入字典
keyword_emb_dict = {
    "keywords": keyword,     # 关键词列表
    "embeddings": embeddings  # 对应的嵌入向量数组
}

# 保存关键词嵌入数据
with open("keyword_embeddings.pkl", "wb") as f:
    pickle.dump(keyword_emb_dict, f)

print("done!")

# ====================================================================
# 输出文件说明
# ====================================================================
# 
# 生成的文件：
# 1. entity_embeddings.pkl - 实体嵌入向量文件
#    包含：{"entities": [实体列表], "embeddings": 嵌入向量数组}
# 
# 2. keyword_embeddings.pkl - 关键词嵌入向量文件  
#    包含：{"keywords": [关键词列表], "embeddings": 嵌入向量数组}
#
# 使用方法：
# import pickle
# with open("entity_embeddings.pkl", "rb") as f:
#     entity_data = pickle.load(f)
#     entities = entity_data["entities"]
#     embeddings = entity_data["embeddings"]

# ====================================================================
# 应用场景
# ====================================================================
# 
# 这些嵌入向量可以用于：
# 1. 实体链接：将问题中的实体与知识图谱中的实体进行匹配
# 2. 语义检索：基于语义相似度查找相关实体
# 3. 知识增强：为问答系统提供相关的背景知识
# 4. 实体消歧：区分同名但不同含义的实体
# 5. 关系预测：基于实体语义预测可能的关系类型
#
# 嵌入向量的优势：
# - 捕获语义信息：相似含义的实体在向量空间中距离更近
# - 支持模糊匹配：可以找到语义相关但表述不同的实体
# - 高效计算：向量运算比文本匹配更快
# - 跨语言支持：预训练模型支持多种语言的嵌入