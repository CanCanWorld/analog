import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    tv = TfidfVectorizer()
    train = ["Chinese Beijing Chinese", "Chinese Chinese Shanghai", "Tokyo Japan Chinese"]
    """
    tf = n/N
        n: N: 
    idf = log(D+1/d+1) + 1
    chinese = 2/3 * [log(3 + 1/3 + 1) + 1] = 2/3
    beijing = 1/3 * [log(4/2) + 1] = 1/3 * (log2+1)
    
    chinese = 2/3
    Shanghai = 1/3 * (log2+1)
    
    Tokyo = 1/3 * (log2+1)
    Japan = 1/3 * (log2+1)
    Chinese = 1/3
        ['beijing' 'chinese' 'japan' 'shanghai' 'tokyo']
    1  1/3 * (log2+1) 2/3
    2               2/3         1/3 * (log2+1)
    3               1/3     1/3 * (log2+1)           1/3 * (log2+1)
    (0,0)  1/3 * (log2+1)   log2 + 1
    (0,1)  2/3              2
    (1,3)  1/3 * (log2+1)   log2 + 1
    (1,1)  2/3              2
    (2,2)  1/3 * (log2+1)   log2 + 1
    (2,4)  1/3 * (log2+1)   log2 + 1
    (2,1)  1/3              1
    """
    tv_fit = tv.fit_transform(train)
    print("特征名称: \n", tv.get_feature_names_out())
    print("tv_fit: \n", tv_fit)

    # fr = open("D:\\project\\python\\analog\\analog\\cache\\model.pkl", 'rb')  # open的参数是pkl文件的路径
    # inf = pickle.load(fr)  # 读取pkl文件的内容
    # print(inf)
    # fr.close()
