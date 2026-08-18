# src/rhyme_utils.py
import pypinyin

# 简单韵母映射（可扩展成古韵表）
def get_rhyme_char(ch):
    """
    返回中文字符韵母
    """
    py = pypinyin.lazy_pinyin(ch, style=pypinyin.Style.FINALS_TONE3)
    if py:
        return py[0]
    return None

def rhyme_score(sentence_list):
    """
    计算诗句押韵率
    sentence_list: list of strings
    返回押韵比例 0-1
    """
    if len(sentence_list) < 2:
        return 0.0
    # 获取每句最后一个字韵母
    rhymes = []
    for sent in sentence_list:
        if len(sent) == 0:
            continue
        rh = get_rhyme_char(sent[-1])
        if rh:
            rhymes.append(rh)
    if len(rhymes) < 2:
        return 0.0
    # 计算押韵比例（最后一句和其他句相同韵母计为押韵）
    last_rhyme = rhymes[-1]
    count = sum(1 for r in rhymes[:-1] if r == last_rhyme)
    return count / (len(rhymes)-1)

def is_rhyme(ch1, ch2):
    """判断两个字是否押韵"""
    r1 = get_rhyme_char(ch1)
    r2 = get_rhyme_char(ch2)
    return r1 == r2 if r1 and r2 else False
