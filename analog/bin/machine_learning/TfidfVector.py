from sklearn.feature_extraction.text import TfidfVectorizer

import os
from analog.bin.lib.utils import read_by_group
import string
from analog.bin.exception.Exceptions import *


class TfidfVector(TfidfVectorizer):
    """
    TF-IDF向量类
    """

    def __init__(self, root_path, config):
        # tf-idf 向量初始化，截取步长为2Bytes
        # super().__init__(smooth_idf=True, use_idf=True, max_df=0.85, min_df=1, lowercase=False,
        #                  vocabulary=self.vocabulary_iter())
        super().__init__(smooth_idf=True, use_idf=True, max_df=0.85, min_df=1, lowercase=False)
        self.fit_vector = None
        self.root_path = root_path
        self.config = config
        self.section_name_log = "Log"
        self.log_path = os.path.join(self.root_path, "analog/sample_set/train.txt")
        self.__fit()

    def __fit(self):
        print('__fit')
        fit_list = []

        if os.path.isfile(self.log_path):
            read_by_group(self.log_path, fit_list,
                          pattern=self.config.get(self.section_name_log, 'log_content_pattern'))
            if len(fit_list) == 0:
                raise FileEmptyError
            self.fit_vector = self.fit_transform(fit_list)
            # print('424055\n', self.get_feature_names_out()[424055])
            # print('252428\n', self.get_feature_names_out()[252428])
            # print('252316\n', self.get_feature_names_out()[252316])
            # print('322614\n', self.get_feature_names_out()[322614])
            # print('271432\n', self.get_feature_names_out()[271432])
            # print('263214\n', self.get_feature_names_out()[263214])
            # print('50303\n', self.get_feature_names_out()[50303])
            # print('40302\n', self.get_feature_names_out()[40302])
            # print('10203\n', self.get_feature_names_out()[10203])
            print("特征名称: \n", self.get_feature_names_out())
            print("特征名称size: \n", self.get_feature_names_out().size)
            print('特征向量化样本\n', self.fit_vector)

        else:
            raise FileNotFound

    @staticmethod
    def vocabulary_iter():
        for i in string.printable:
            for j in string.printable:
                yield i + j
