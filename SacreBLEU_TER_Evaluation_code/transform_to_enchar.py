#coding=utf-8
import sys

def to_en_word(line1, line2):
    words1 = line1.split()
    words2 = line2.split()
    allword = words1 + words2
    cn2en = {'。':'.'}
    cn2en = {}
    # en2cn = {}
    num = 0
    for word in allword:
        if word not in cn2en:
            cn2en[word] = 'word{}'.format(num)
            # en2cn['word{}'.format(num)] = word
            num += 1
    words1_new = [cn2en[word] for word in words1]
    words2_new = [cn2en[word] for word in words2]
    return ' '.join(words1_new), ' '.join(words2_new)


print("transform test", sys.argv[1], sys.argv[2])
with open(sys.argv[1], 'r', encoding='utf-8') as file1, open(sys.argv[2], 'r', encoding='utf-8') as file2, \
     open(sys.argv[1] + '.en', 'w', encoding='utf-8') as file3, open(sys.argv[2] + '.en', 'w', encoding='utf-8') as file4:
    for line1, line2 in zip(file1, file2):
        line1, line2 = line1.strip(), line2.strip()
        line1, line2 = to_en_word(line1, line2)
        file3.write(line1 + '\n')
        file4.write(line2 + '\n')
