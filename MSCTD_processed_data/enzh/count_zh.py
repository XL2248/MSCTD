#coding=utf-8
import os, sys
import string
import code
def str_count(str):
    '''找出字符串中的中英文、空格、数字、标点符号个数'''
    count_en = count_dg = count_sp = count_zh = count_pu = 0

    for s in str:
        # 英文
        if s in string.ascii_letters:
            count_en += 1
        # 数字
        elif s.isdigit():
            count_dg += 1
        # 空格
        elif s.isspace():
            count_sp += 1
        # 中文
        elif s.isalpha():
            count_zh += 1
        # 特殊字符
        else:
            count_pu += 1
    return count_en + count_dg + count_sp + count_zh + count_pu

word_num = 0
utt_num = 0 
d = {}
file=sys.argv[1]
with open(file, 'r', encoding='utf-8') as fr:
#for d_id, dialog in enumerate(group_ids):
    for item in fr.readlines():
#    dialog_num += 1
        utt_num += 1
        word_num += str_count(item.strip())
    # for u_id, item in enumerate(dialog):
print(utt_num, word_num, len(set(d)), d)
#code.interact(local=locals())
