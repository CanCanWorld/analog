from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

if __name__ == '__main__':
    # 读数据
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    # 标准化
    std = StandardScaler()
    x_std = std.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=0.3)

    svm = SVC()  # 分类
    # svm = SVR()  # 回归
    svm.fit(x_train, y_train)

    score = svm.score(x_test, y_test)
    print('score :', score)
