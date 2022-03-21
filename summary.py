import gzip
import json
import pandas as pd
import os.path
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
import gensim.downloader


import nltk
nltk.download('punkt')
nltk.download('stopwords')

def ini():
    assert os.path.exists('All_Beauty_5.json.gz') and os.path.exists('meta_All_Beauty.json'), 'no such files'
    def getDF(path):
        def parse(path):
            g = gzip.open(path, 'rb')
            for l in g:
                yield json.loads(l)
        i = 0
        df = {}
        for d in parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    df_1 = getDF('All_Beauty_5.json.gz')
    df_2 = pd.read_json('meta_All_Beauty.json', lines=True)
    return df_1, df_2

class Preprocess:
    def __init__(self, ui_df: pd.DataFrame, item_df: pd.DataFrame):
        self.ui_df = ui_df
        self.item_df = item_df
        self.training_data, self.test_data = self.__split_ui_data()
        self.clean_item_data = self.__clean_item_data()
        self.ui_matrix = self.__ui_matrix_training()

    class Prediction:
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
    def __init__(self, ui_df, item_df):
        super().__init__(ui_df, item_df)

    def cos_sim_ui(self, user_id: str, item_id: str):
        cosine_similarities = cosine_similarity(np.array(
            self.ui_matrix.loc[self.ui_matrix.index == user_id]), self.ui_matrix[self.ui_matrix.loc[:, item_id] > 0])
        result = pd.DataFrame(
            self.ui_matrix[self.ui_matrix.loc[:, item_id] > 0].loc[:, item_id])
        result.insert(result.shape[1], 'cos_sim',
                      np.array(cosine_similarities).T)
        return result

    def cos_sim_prediction(self, user_id: str, item_id: str, rank: int):
        result = self.cos_sim_ui(
            user_id, item_id)
        result_sort = result.sort_values(
            by='cos_sim', ascending=False).head(rank)
        prediction = 0
        for index in result_sort.index:
            prediction += result_sort.loc[index][0]*result_sort.loc[index][1]
        prediction /= result_sort.loc[:, 'cos_sim'].sum()
        return prediction

    def simple_svd(self, user_id: str, item_id: str, k: int, mute: bool):
        matrix = self.ui_matrix
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
    def __init__(self, prediction_list, ui_df, item_df):
        super().__init__(ui_df, item_df)
        self.prediction_list = prediction_list
        self.clean_pred_list = self.__clean_pred_list()
        self.ui_relevant_matrix = self.__ui_relevant_matrix()

    def __clean_pred_list(self):
        user_in_prediction = set([pred.uid for pred in self.prediction_list])
        users_in_pred_but_not_in_test = list(
            user_in_prediction.difference(set(self.test_data['reviewerID'])))
        out_list = []
        for i, name in enumerate(self.prediction_list):
            if name.uid not in users_in_pred_but_not_in_test:
                out_list.append(self.prediction_list[i])
        # print(len(out_list))
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
        num_users = len(self.ui_relevant_matrix.index)
        summation = []
        user_item_rating = defaultdict(list)
        for pred in self.prediction_list:
            user_item_rating[pred.uid].append((pred.iid, pred.est))
        for user_id, filted_pred_list in user_item_rating.items():
            filted_pred_list.sort(key=lambda x: x[1], reverse=True)
            summation.append(function_(filted_pred_list, user_id,
                             cut_off))
        return sum(summation)/float(num_users)

    def rank_k(self)->dict:
        user_item_rating = defaultdict(list)
        for pred in self.prediction_list:
            user_item_rating[pred.uid].append((pred.iid, pred.est))
        for user_id, filted_pred_list in user_item_rating.items():
            filted_pred_list.sort(key=lambda x: x[1], reverse=True)
        return user_item_rating


class Contentbased_recommendation(Preprocess):

    def __init__(self, ui_df, item_df, word2vec:bool=False):
        super().__init__(ui_df, item_df)
        self.original_title = list(self.clean_item_data['title'])
        self.clean_title = self.__clean_title()
        if word2vec:
            self.tfidf = self.__tfidf_by_word2vec()
        else:
            self.tfidf = self.__tfidf()
        self.user_mean_vector = self.__user_mean_vector()

    def __clean_title(self):
        porter_stemmer = PorterStemmer()
        title_list = []
        for title in self.original_title:
            filter_list = [porter_stemmer.stem(word.lower()) for word in word_tokenize(
                title) if word not in stopwords.words("english") and word.isalpha()]
            title_list.append(
                TreebankWordDetokenizer().detokenize(filter_list))
        return title_list

    def __tfidf(self):
        tfidf_vectorizer = TfidfVectorizer()
        x = tfidf_vectorizer.fit_transform(self.clean_title)
        return pd.DataFrame(x.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=self.clean_item_data['asin'])

    def __user_mean_vector(self):
        user_mean_vector_dic = {}
        # print(user_list)
        for uid in set(self.training_data['reviewerID']):
            item_in = self.clean_item_data['asin'][self.clean_item_data['asin'].isin(
                self.training_data[self.training_data['reviewerID'] == uid]['asin'])]
            summation = 0
            temp_list = []
            for iid in item_in:
                temp_list.append(self.tfidf.loc[iid,:].to_numpy()*int(self.ui_matrix.loc[uid, iid]))
                summation += int(self.ui_matrix.loc[uid, iid])
            user_mean_vector_dic[uid] = np.sum(temp_list, axis=0)/summation        
        return user_mean_vector_dic

    def pick_item_from_tfidf(self, item_list:list):
            vector_list = []
            for item in item_list:
                vector_list.append(self.tfidf.loc[item,:].to_numpy())
            return vector_list
    
    def prediction(self):
        x = lambda bools: True if bools == False else False
        prediction_list = []
        # print(user_list)
        # chick_1 = 0
        for uid in set(self.training_data['reviewerID']):
            item_not_in = self.clean_item_data['asin'][[x(bools) for bools in self.clean_item_data['asin'].isin(
                self.training_data[self.training_data['reviewerID'] == uid]['asin'])]]
            # chick_1 += len(item_not_in)
            vector_not_in = self.pick_item_from_tfidf(item_not_in)
            cos_sim = cosine_similarity(vector_not_in,[self.user_mean_vector[uid]])
            for i,item in enumerate(item_not_in):
                prediction_list.append(self.Prediction(uid=uid, iid=item, est=cos_sim[i]))
        # print(chick_1)
        # print(len(prediction_list))
        return prediction_list

    def __tfidf_by_word2vec(self):
        word2vec_vectors = gensim.downloader.load('word2vec-google-news-300')
        sentences = [list(word_tokenize(title)) for title in self.clean_title]
        vector_list = []
        for sentence in sentences:
            vector = np.zeros((300,))
            for word in sentence:
                try:
                    vector += word2vec_vectors[word]
                except KeyError :
                    continue
            vector_list.append(vector/len(sentence))
        return pd.DataFrame(np.array(vector_list),index=self.clean_item_data['asin'])




if __name__ == '__main__':
    import gzip
    import json
    import pandas as pd

    def getDF(path):
        def parse(path):
            g = gzip.open(path, 'rb')
            for l in g:
                yield json.loads(l)
        i = 0
        df = {}
        for d in parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    df_1 = getDF('All_Beauty_5.json.gz')
    df_2 = pd.read_json('meta_All_Beauty.json', lines=True)
    preprocess = Contentbased_recommendation(ui_df=df_1, item_df=df_2)
    prediction_list = preprocess.prediction()
    evaluation = Evaluation(prediction_list, df_1, df_2)
    kr_mean = evaluation.mean_k(5, 'hr_k')
    print(evaluation.rank_k()['A39WWMBA0299ZF'][:5])
    print(kr_mean)