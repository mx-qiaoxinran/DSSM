import args

def n_gram(word,n=args.N):
    s=[]
    word='#'+word+'#'
    for i in range(len(word)-2):
        s.append(word[i:i+3])
    return s

def lst_gram(lst,n=args.N):
    s=[]
    for word in str(lst).lower().split():
        s.extend(n_gram(word))
    return s

vocab=[]  #一个存放n-gram切片的列表
file_path='./MRPC/'  #数据集根目录
files=['train_data.csv','test_data.csv']  #训练集、测试集

for file in files:
    f=open(file_path+file,encoding='utf-8').readlines()
    for i in range(1,len(f)):  #从1遍历因为下标0为表头（gold_label sentence1 sentence2）
        s1,s2=f[i][2:].strip('\n').split('\t')
        #对每一行，去掉前两个字符{标签（0，1，2），TAB键}
        #去掉末尾换行符，用TAB键切分字符串，刚好切为两个句子
        vocab.extend(lst_gram(s1))
        vocab.extend(lst_gram(s2))

vocab=set(vocab)
vocab_list = ['[PAD]', '[UNK]']
vocab_list.extend(list(vocab))

vocab_file=args.VOCAB_FILE
with open(vocab_file,'w',encoding='utf-8') as f:
    for slice in vocab_list:
        f.write(slice)
        f.write('\n')