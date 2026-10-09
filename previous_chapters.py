# 本段代码来自 github rasbt/LLMs-from-scratch/ch05/07_gpt_to_llama/ ，版权信息如下

# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# 仅进行必要的中文注释
import torch

def text_to_token_ids(text, tokenizer):
    """将原始文本信息按照 tokenizer 的分次法词元化，同时使用 tokenizer 的词表转换成 int 索引"""
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """
    将 int 索引构成的列表，转换成文本
    与 text_to_token_ids 构成逆操作
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    使用 model 进行文本生成
    model: 大语言模型
    idx: 当前文本的索引列表，(batch_size, num_tokens)，num_tokens 随推理进行变化
    max_new_tokens: 一次 generate 调用最多生成的 token 数量
    context_size: 推导新文本时给 model 的上下文长度
    """
    # 每次循环专注于生成最新一步的词元 logits
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]   # 生成最新一步词元，看倒数 context_size 的上下文窗口
        # 切片特性可以保证：序列还没达到最大上下文窗口 context_size 时，就使用当前全部历史（从 0 切片）。
        with torch.no_grad():
            logits = model(idx_cond)
        # 模型输出 (batch_size, num_tokens, vocab_size)
        logits = logits[:, -1, :]   # 取模型输出的最新的 logits，即输入的末尾位置对应的输出

        # 如果使用 top_k，在输出 logits 中将只保留前 k 个最大值
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            # 将其他 logits 全部置为负无穷，保证 softmax 后为 0
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # 可选进行温度缩放：控制 softmax 分布的“尖锐程度”（方差）
        # temperature < 1：分布更尖锐，高概率 token 更容易被选中，输出更确定。
        # temperature > 1：分布更平坦，低概率 token 也有机会被选中，输出更随机。
        # temperature = 1：就是原始 softmax 分布。
        if temperature > 0.0:
            # 改变尖锐程度，通过 scaling 实现
            logits = logits / temperature
            # 减去行最大值，不改变分布形状，只是保证数值稳定性
            logits = logits - logits.max(dim=-1, keepdim=True).values
            # softmax 获取概率值
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)
            # 依上面得到的概率分布，进行随机取样输出
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        # 温度为 0 直接跳过缩放和依概率分布取输出，直接输出 logits 最大值的索引
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # 如果输出了特殊的 end-of-sequence token 的索引，终止输出
        if idx_next == eos_id:
            break

        # 将最新的输出连接到 idx 最后
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens + 1)

    return idx
