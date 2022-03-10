#coding=utf-8
import os, sys, code
import random

start = 0 #int(sys.argv[1])
end = 1 #int(sys.argv[2])
typ = sys.argv[1]
filepath_w_en = './'+typ+'_en.txt'
filepath_w_ch = './'+typ+'_de.txt'
filepath_w_en_ctx = './'+typ+'_en_ctx.txt'
filepath_w_ch_ctx = './'+typ+'_de_ctx.txt'
filepath_w_en_ctx_src = './'+typ+'_en_ctx_src.txt'
filepath_w_ch_ctx_src = './'+typ+'_de_ctx_src.txt'
filepath_w_chen_ctx = './'+ typ+'_en_sample.txt'
filepath_w_chper_ctx = './'+ typ+'_de_sample.txt'

f_en = open(filepath_w_en, 'w', encoding='utf-8')
f_ch = open(filepath_w_ch, 'w', encoding='utf-8')
f_en_ctx = open(filepath_w_en_ctx, 'w', encoding='utf-8')
f_ch_ctx = open(filepath_w_ch_ctx, 'w', encoding='utf-8')
f_en_sam = open(filepath_w_chen_ctx, 'w', encoding='utf-8')
f_ch_sam = open(filepath_w_chper_ctx, 'w', encoding='utf-8')
f_ctx_src_en = open(filepath_w_en_ctx_src, 'w', encoding='utf-8')
f_ctx_src_ch = open(filepath_w_ch_ctx_src, 'w', encoding='utf-8')
path = "./"
for idx in range(start, end):
#    en_filename = "english_%s.txt.norm.tok.bpe.10k"%typ
#    zh_filename = "german_%s.txt.norm.tok.bpe.10k"%typ
    zh_filename = "%s.tok.bpe.32000.de"%typ
    en_filename = "%s.tok.bpe.32000.en"%typ
    en_content, zh_content = [], []
    if os.path.exists(path + en_filename):
        with open(path + en_filename, 'r', encoding='utf-8') as fr:
           for line in fr.readlines():
               en_content.append(" ".join(line.strip().replace('\t', ' ').replace('\n', ' ').replace('\t\n', ' ').split()))

    if os.path.exists("./" + zh_filename):
        with open("./" + zh_filename, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
               zh_content.append(" ".join(line.strip().replace('\t', ' ').replace('\n', ' ').replace('\t\n', ' ').split()))

    assert len(en_content) == len(zh_content)
    idx = -1
    s = 0
    for en_line, zh_line in zip(en_content, zh_content):
        idx += 1
        en_ctx, zh_ctx, en_sam, zh_sam = [], [], '', ''
        if idx == 0:
            en_ctx, zh_ctx = [""], [""]
            en_rest = en_content[1:] #list(set(all_source)^set(Source)) #Source - switch_context
            zh_rest = zh_content[1:]
            sam_num = random.randint(0, (len(en_rest)-2))
            en_sam = en_rest[sam_num]
            sam_num = random.randint(0, (len(zh_rest)-2))
            zh_sam = zh_rest[sam_num]
#            code.interact(local=locals())
        elif idx == 1:
            en_ctx.append(en_content[0])
            zh_ctx.append(zh_content[0]) 
            en_sam = en_content[0]
            zh_sam = zh_content[0]
#            code.interact(local=locals())
        elif idx == 2:
            en_ctx.append(en_content[0])
            zh_ctx.append(zh_content[0])
            en_ctx.append(en_content[1])
            zh_ctx.append(zh_content[1])

            en_rest = en_content[0:2] #list(set(all_source)^set(Source)) #Source - switch_context
            zh_rest = zh_content[0:2]
            sam_num = random.randint(0, (len(en_rest)-1))
            en_sam = en_rest[sam_num]
            sam_num = random.randint(0, (len(zh_rest)-1))
            zh_sam = zh_rest[sam_num]
#            code.interact(local=locals())
        else:
            en_ctx.append(en_content[idx - 3])
            zh_ctx.append(zh_content[idx - 3])
            en_ctx.append(en_content[idx - 2])
            zh_ctx.append(zh_content[idx - 2])
            en_ctx.append(en_content[idx - 1])
            zh_ctx.append(zh_content[idx - 1])
         
            if idx > 10:
                s = idx - 10
            else:
                s = 0
            en_rest = en_content[s:idx] #list(set(all_source)^set(Source)) #Source - switch_context
            zh_rest = zh_content[s:idx]
            sam_num = random.randint(0, (len(en_rest)-1))
            en_sam = en_rest[sam_num]
            sam_num = random.randint(0, (len(zh_rest)-1))
            zh_sam = zh_rest[sam_num]
#            code.interact(local=locals())

        f_en.write(en_line + '\n') #continue;
        f_ch.write(zh_line + '\n')
        f_en_ctx.write("[CLS] " + " [SEP] ".join(en_ctx) + " [SEP]\n")
        f_ch_ctx.write("[CLS] " + " [SEP] ".join(zh_ctx) + " [SEP]\n")
        f_ctx_src_en.write("[CLS] " + " [SEP] ".join(en_ctx) + " [SEP] " + en_line + " <eos>\n")
        f_ctx_src_ch.write("[CLS] " + " [SEP] ".join(zh_ctx) + " [SEP] " + zh_line + " <eos>\n")
        f_en_sam.write(en_sam + '\n')
        f_ch_sam.write(zh_sam + '\n')
#    code.interact(local=locals())

f_en.close()
f_ch.close()
f_en_ctx.close()
f_ch_ctx.close()
f_en_sam.close()
f_ch_sam.close()
f_ctx_src_en.close()
f_ctx_src_ch.close()
print("DONE!")
