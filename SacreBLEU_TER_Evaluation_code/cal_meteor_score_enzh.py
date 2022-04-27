#coding=utf-8

import os, sys
#checkpoint_step=sys.argv[1].split(',')
hyp=sys.argv[1] # model name
start=int(sys.argv[2])
end=int(sys.argv[3])
#for idx in checkpoint_step:
for idx in range(start, end, 1000):
    tmp = hyp + '/test.out.zh.delbpe.char.' + str(idx)
    os.system("python transform_to_enchar.py /path/to/generated/file/%s /path/to/reference/file"%tmp) # character-level for Chinese
    os.system("java -Xmx2G -jar meteor-*.jar /path/to/transformed/generated/file/%s.en /path/to/transformed/reference/file/test.tok.zh.char.en -norm -writeAlignments -f system1"%tmp)
