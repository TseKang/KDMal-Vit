# prompt.py
import torch
import torch.nn as nn

class Prompt(nn.Module):
    def __init__(self, num_classes, prompt_length, embed_dim):
        super(Prompt, self).__init__()
        self.num_classes = num_classes
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        # 为每个类别定义可学习的Prompt参数
        self.prompt_embeddings = nn.Parameter(torch.randn(num_classes, prompt_length, embed_dim))

    def forward(self, class_indices):
        # 根据输入的类别索引选择对应的Prompt
        selected_prompts = self.prompt_embeddings[class_indices]
        return selected_prompts