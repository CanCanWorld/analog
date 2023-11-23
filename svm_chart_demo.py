import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # 选取前两个特征，为了可视化目的
    y = iris.target

    # 我们创建一个 SVM 分类器
    C = 1.0  # SVM 正则化参数
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)

    # 创建一个图，画出数据点以及决策边界
    # 为了画图的目的，我们只对特征空间做一个网格，然后画出决策边界。
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = .02  # 网格步长
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

    # 将结果放入彩色图中
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # 画出训练数据
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('SVC with linear kernel')
    plt.show()