import json

from torch.utils.data import Dataset
import torch
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # 关闭 tokenizers 的并行化, 避免并行带来的死锁问题

# 定义一个简单的语言模型数据集类，用 transformers 的 PreTrainDataset 来加载数据

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, seq_len=521): # 单次可以处理的最大长度
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = seq_len
        self.samples = self.load_data(data_path) # 加载数据, 返回一个列表, 储存所有样本
    
    def load_data(self, data_path):
        # 从指定路径加载数据
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f: # 只读打开，编码方式 utf-8
            for i, line in enumerate(f, 1): # enumerate 指定 i 从 1 开始计数
                data = json.loads(line.strip()) # 逐行读取并解析为 JSON 对象
                samples.append(data)
        return samples

    def __len__(self):
        # 返回数据集的样本数量
        return len(self.samples)

    # 根据索引获取样本
    def __getitem__(self, idx):
        # 获取指定索引的样本
        sample = self.samples[idx]
        encoding = self.tokenizer(
            str(sample['text']), # 将文本转换为字符串
            max_length=self.max_length,
            padding='max_length', # 填充到最大长度
            truncation=True, # 超过最大长度则截断
            return_tensors='pt' # 返回 PyTorch 张量
        )

        input_ids = encoding['input_ids'].squeeze(0) # 去掉批次维度
        loss_mask = input_ids != self.tokenizer.pad_token_id # 将填充部分标记为不计算损失
        
        # 自回归
        x = input_ids[:-1].clone()  # 输入序列，去掉最后一个 token
        y = input_ids[1:].clone()   # 目标序列，去掉第一个 token
        
        # loss_mask 和 y 对齐
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.float) # 转换为浮点型张量

        return x, y, loss_mask