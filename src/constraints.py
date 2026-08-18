# src/constraints.py
#import torch

def apply_acrostic_constraint(batch_output_ids, head_chars_ids):
    """
    batch_output_ids: list of generated ids (B, T)
    head_chars_ids: list of ids to放在每句开头
    返回约束后的输出
    """
    constrained = []
    for out_ids in batch_output_ids:
        new_ids = out_ids[:]
        # 假设每句长度相同，用简单方式，每句开头替换
        for i, hid in enumerate(head_chars_ids):
            if i < len(new_ids):
                new_ids[i] = hid
        constrained.append(new_ids)
    return constrained

def apply_tail_constraint(batch_output_ids, tail_chars_ids):
    """类似藏尾，保证每句末尾为指定字符"""
    constrained = []
    for out_ids in batch_output_ids:
        new_ids = out_ids[:]
        for i, tid in enumerate(tail_chars_ids):
            if i < len(new_ids):
                new_ids[-len(tail_chars_ids)+i] = tid
        constrained.append(new_ids)
    return constrained

def apply_palindrome_constraint(ids):
    """简单回文约束"""
    half_len = len(ids) // 2
    new_ids = ids[:half_len] + ids[:half_len][::-1]
    if len(ids) % 2 == 1:
        new_ids.append(ids[half_len])
    return new_ids
