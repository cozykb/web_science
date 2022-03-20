from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from surprise import Reader
from surprise import Dataset
from surprise import SVD
from surprise.model_selection import GridSearchCV
from surprise import KNNWithMeans
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


import nltk
nltk.download('punkt')
nltk.download('stopwords')


class Preprocess:
    def __init__(self, ui_df: pd.DataFrame, item_df: pd.DataFrame):
        self.ui_df = ui_df
        self.item_df = item_df
        self.training_data, self.test_data = self.__split_ui_data()
        self.clean_item_data = self.__clean_item_data()
        self.ui_matrix = self.__ui_matrix_training

    class Predection:
        def __init__(self, uid, iid, est):
            self.uid = uid
            self.iid = iid
            self.est = est

    def __split_ui_data(self):
        df = self.ui_df.sort_values(
            by=['reviewerID', 'asin', 'unixReviewTime'])
        cleaned_dataset = df.dropna(subset=['overall']).drop_duplicates(
            subset=['reviewerID', 'asin'], keep='last').reset_index(drop=True)
        cleaned_dataset = cleaned_dataset.sort_values(
            by=['reviewerID', 'unixReviewTime']).reset_index(drop=True)
        test_data_pre = cleaned_dataset[cleaned_dataset.overall >= 4.0].drop_duplicates(
            subset=['reviewerID'], keep='last')
        training_data = cleaned_dataset.drop(test_data_pre.index)
        user_in_training = test_data_pre['reviewerID'].isin(
            training_data['reviewerID'])
        test_data = test_data_pre[user_in_training]
        return training_data, test_data

    def __clean_item_data(self):
        df = self.item_df
        df = df.sort_values(by=['asin'])
        clean_dataset_item = df.drop_duplicates(
            subset=['asin'], keep='last').reset_index(drop=True)
        item_in_subset = list(
            self.test_data.loc[:, 'asin'])+list(self.training_data.loc[:, 'asin'])
        clean_dataset_item = clean_dataset_item[clean_dataset_item['asin'].isin(
            item_in_subset)]
        return clean_dataset_item

    def __ui_matrix_training(self):
        matrix = pd.DataFrame(columns=self.training_data['asin'].drop_duplicates(
        ), index=self.training_data['reviewerID'].drop_duplicates())
        for row_i in range(len(self.training_data)):
            reviewerID = list(self.training_data.reviewerID)[row_i]
            asin = list(self.training_data.asin)[row_i]
            rate = list(self.training_data.overall)[row_i]
            matrix.loc[reviewerID, asin] = rate
        matrix = matrix.fillna(0)
        return matrix


class Collaborativefiltering_recommendation(Preprocess):

    def cos_sim_ui(self, user_id: str, item_id: str):
        cosine_similarities = cosine_similarity(np.array(
            self.ui_matrix.loc[self.ui_matrix.index == user_id]), self.ui_matrix[self.ui_matrix.loc[:, item_id] > 0])
        result = pd.DataFrame(
            self.ui_matrix[self.ui_matrix.loc[:, item_id] > 0].loc[:, item_id])
        result.insert(result.shape[1], 'cos_sim',
                      np.array(cosine_similarities).T)
        return result

    def cos_sim_prediction(self, user_id: str, item_id: str, rank: int):
        result = Collaborativefiltering_recommendation.cos_sim_ui(
            user_id, item_id)
        result_sort = result.sort_values(
            by='cos_sim', ascending=False).head(rank)
        prediction = 0
        for index in result_sort.index:
            prediction += result_sort.loc[index][0]*result_sort.loc[index][1]
        prediction /= result_sort.loc[:, 'cos_sim'].sum()
        return prediction

    def simple_svd(self, user_id: str, item_id: str, k: int, mute: bool):
        matrix = pd.DataFrame(columns=self.training_data['asin'].drop_duplicates(
        ), index=self.training_data['reviewerID'].drop_duplicates())
        for row_i in range(len(self.training_data)):
            reviewerID = list(self.training_data.reviewerID)[row_i]
            asin = list(self.training_data.asin)[row_i]
            rate = list(self.training_data.overall)[row_i]
            matrix.loc[reviewerID, asin] = rate
        matrix = matrix.fillna(0)
        for index in matrix.index:
            mean_ = np.mean([i for i in matrix.loc[index] if i != 0])
            if index == user_id:
                user_id_mean = mean_
            for item in range(len(matrix.loc[index])):
                if matrix.loc[index][item] != 0:
                    matrix.loc[index][item] -= mean_
        Q, sigma, P = svds(matrix, k)
        U = np.dot(Q, np.diag(sigma))
        user_factors = pd.DataFrame(data=U, index=matrix.index)
        item_factors = pd.DataFrame(data=P, columns=matrix.columns)
        if not mute:
            print(user_factors.loc[user_id])
            print(item_factors.loc[:, item_id])
        prediction = user_id_mean + \
            np.dot(np.array(user_factors.loc[user_id]), np.array(
                item_factors.loc[:, item_id]))
        return prediction

    def gridsearch(self, method: str, search_array: list, measures_method=['rmse'], cv_num=3):
        reader = Reader(rating_scale=(1, 5))
        training = Dataset.load_from_df(
            self.training_data[['reviewerID', 'asin', 'overall']], reader=reader)
        assert method == 'knn' or method == 'svd', 'method isn\'t knn or svd'
        if method == 'knn':
            param_grid = {'k': search_array,
                          'sim_options': {'name': ['cosine'],
                                          # compute  similarities between items
                                          'user_based': [True]
                                          }
                          }
            gs = GridSearchCV(KNNWithMeans, param_grid,
                              measures=measures_method, cv=cv_num)
        if method == 'svd':
            param_grid = {'n_epochs': search_array, 'n_factors': [30]}
            gs = GridSearchCV(
                SVD, param_grid, measures=measures_method, cv=cv_num)
        return gs.fit(training)

    def knn(self, k=10, sim_options={'name': 'cosine', 'user_based': True}):
        reader = Reader(rating_scale=(1, 5))
        training = Dataset.load_from_df(
            self.training_data[['reviewerID', 'asin', 'overall']], reader=reader)
        trainset = training.build_full_trainset()
        testset = trainset.build_anti_testset()
        model = KNNWithMeans(k=k, sim_options=sim_options)
        return model.fit(trainset).test(testset)

    def svd(self, n_epochs=500, n_factors=30, random_state=0):
        reader = Reader(rating_scale=(1, 5))
        training = Dataset.load_from_df(
            self.training_data[['reviewerID', 'asin', 'overall']], reader=reader)
        trainset = training.build_full_trainset()
        testset = trainset.build_anti_testset()
        model = SVD(n_epochs=n_epochs, n_factors=n_factors,
                    random_state=random_state)
        return model.fit(trainset).test(testset)


class Evaluation(Preprocess):
    def __init__(self, predection_list):
        super().__init__()
        self.predection_list = predection_list
        self.clean_pred_list = self.__clean_pred_list()
        self.ui_relevant_matrix = self.__ui_relevant_matrix()

    def __clean_pred_list(self):
        user_in_prediction = set([pred.uid for pred in self.predection_list])
        users_in_pred_but_not_in_test = list(
            user_in_prediction.difference(set(self.test_data['reviewerID'])))
        out_list = []
        for i, name in enumerate(self.predection_list):
            if name.uid not in users_in_pred_but_not_in_test:
                out_list.append(self.predection_list[i])
        return out_list

    def top_prediction(self, rank, uid):
        filted_pred_list = list(
            filter(lambda x: x.uid == uid, self.clean_pred_list))
        filted_pred_list.sort(key=lambda x: x.est, reverse=True)
        return [(i.iid, i.est) for i in filted_pred_list][:rank]

    def __ui_relevant_matrix(self):
        relevant_matrix = pd.DataFrame(columns=self.test_data['asin'].drop_duplicates(
        ), index=self.test_data['reviewerID'].drop_duplicates())
        for row_i in range(len(self.test_data)):
            reviewerID = list(self.test_data.reviewerID)[row_i]
            asin = list(self.test_data.asin)[row_i]
            rate = list(self.test_data.overall)[row_i]
            relevant_matrix.loc[reviewerID, asin] = 1 if rate >= 4 else 0
        relevant_matrix = relevant_matrix.fillna(0)
        return relevant_matrix

    def p_k_user(self, filted_pred_list, user_id, cut_off):
        summation = []
        for i in range(cut_off):
            item_id = filted_pred_list[i][0]
            try:
                summation.append(self.ui_relevant_matrix.loc[user_id, item_id])
            except KeyError:
                summation.append(0)
        return sum(summation)/float(cut_off)

    def ap_k_user(self, filted_pred_list, user_id, cut_off):
        num_relevance = sum(self.ui_relevant_matrix.loc[user_id, :])
        summation = []
        for i in range(cut_off):
            item_id = filted_pred_list[i][0]
            try:
                if self.ui_relevant_matrix.loc[user_id, item_id] == 1:
                    summation.append(self.p_k_user(
                        filted_pred_list, user_id, i+1, self.ui_relevant_matrix))
            except KeyError:
                summation.append(0)
        # if num_relevance == 0:
        #     return 0
        return sum(summation)/float(num_relevance)

    def rr_k_user(self, filted_pred_list, user_id, cut_off):
        for i in range(cut_off):
            item_id = filted_pred_list[i][0]
            try:
                if self.ui_relevant_matrix.loc[user_id, item_id] == 1:
                    return 1/float(i+1)
            except KeyError:
                continue
        return 0

    def hr_k_user(self, filted_pred_list, user_id, cut_off):
        for i in range(cut_off):
            item_id = filted_pred_list[i][0]
            try:
                if self.ui_relevant_matrix.loc[user_id, item_id] == 1:
                    return 1
            except KeyError:
                continue
        return 0

    def mean_k(self, cut_off, function_name: str):
        assert function_name in [
            'p_k', 'ap_k', 'rr_k', 'hr_k'], "function name not in ['p_k', 'ap_k', 'rr_k', 'hr_k']"
        switch = {
            'p_k': self.p_k_user,
            'ap_k': self.ap_k_user,
            'rr_k': self.rr_k_user,
            'hr_k': self.hr_k_user
        }
        function_ = switch[function_name]
        user_list = []
        user_list = [
            item.uid for item in self.predection_list if item.uid not in user_list]
        num_users = len(self.ui_relevant_matrix.index)
        summation = []
        user_item_rating = defaultdict(list)
        for pred in self.predection_list:
            user_item_rating[pred.uid].append((pred.iid, pred.est))
        for user_id, filted_pred_list in user_item_rating.items():
            filted_pred_list.sort(key=lambda x: x[1], reverse=True)
            summation.append(function_(filted_pred_list, user_id,
                             cut_off, self.ui_relevant_matrix))
        return sum(summation)/float(num_users)


class Contentbased_recommendation(Preprocess):
    def __init__(self):
        super().__init__()
        self.tfidf = self.__tfidf()
        self.original_title = list(self.clean_item_data['title'])
        self.clean_title = self.__clean_title()

    def __clean_title(self):
        porter_stemmer = PorterStemmer()
        title_list = []
        for title in self.original_title:
            filter_list = [porter_stemmer.stem(word) for word in word_tokenize(
                title) if word not in stopwords.words("english")]
            title_list.append(
                TreebankWordDetokenizer().detokenize(filter_list))
        return title_list

    def __tfidf(self):
        tfidf_vectorizer = TfidfVectorizer()
        x = tfidf_vectorizer.fit_transform(self.clean_title)
        return pd.DataFrame(x.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=self.clean_item_data['asin'])

    def __predection(self):
        def x(bools): return True if bools == False else False
        for uid in self.training_data['reviewerID']:
            item_in = self.clean_item_data['asin'][self.clean_item_data['asin'].isin(
                self.training_data[self.training_data['reviewerID'] == uid]['asin'])]
            item_not_in = self.clean_item_data['asin'][[x(bools) for bools in self.clean_item_data['asin'].isin(
                self.training_data[self.training_data['reviewerID'] == uid]['asin'])]]
