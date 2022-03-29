import nltk
from transformers import BertModel, BertTokenizerFast
import gzip
import json
import pandas as pd
import os.path
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from surprise import Reader
from surprise import Dataset
from surprise import SVD
import surprise
from surprise.model_selection import GridSearchCV
from surprise import KNNWithMeans
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader
import copy
import torch
import transformers
assert transformers.__version__ > '4.0.0'
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TOKENIZERS_PARALLELISM = 'false'
print(f"Using device: {DEVICE}")


nltk.download('punkt')
nltk.download('stopwords')


def ini(path_1, path_2):
    # ini function is used to create the pandas Dataframe according to the input json files.
    # it allows the input in form *.json and *.gz
    # Make sure the input files are in the same directory as this file, or enter the correct absolute location
    assert os.path.exists(path_1) and os.path.exists(path_2), 'no such files'

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
    if path_1[-2:] == 'gz':
        df_1 = getDF(path_1)
    else:
        df_1 = pd.read_json(path_1, lines=True)
    if path_2[-2:] == 'gz':
        df_2 = getDF(path_2)
    else:
        df_2 = pd.read_json(path_2, lines=True)
    return df_1, df_2


class Preprocess:
    # this class is used to clean and split the data into training and test set
    # you need to initial the class with user item Dataframe and meta Dataframe
    # self.ui_df is the original user item Dataframe
    # self.item_df is the original meta item Dataframe
    # self.training_data is the training set and self.test_data is the test set, these data sets are made with cleaned data
    # self.clean_item_data is the cleaned meta item Data
    # self.ui_matrix is the user-item matrix according to training set
    def __init__(self, ui_df: pd.DataFrame, item_df: pd.DataFrame):
        self.ui_df = ui_df
        self.item_df = item_df
        self.training_data, self.test_data = self.__split_ui_data()
        self.clean_item_data = self.__clean_item_data()
        self.ui_matrix = self.__ui_matrix_training()

    class Prediction:
        # this class is used to form a standard prediction list for further evaluation
        # self.uid is the user ID
        # self.iid is the item ID
        # self.est is the prediction rating score
        def __init__(self, uid, iid, est):
            self.uid = uid
            self.iid = iid
            self.est = est

    def __split_ui_data(self):
        # this private function is used to clean and split the data
        # Sort by review time, delete data rows without ratings, and remove duplicate
        # for each user, find the data with the latest rating score >= 4, extract this data to form the test set
        # remain data will be the training set
        df = self.ui_df.sort_values(
            by=['reviewerID', 'asin', 'unixReviewTime'])
        cleaned_dataset = df.dropna(subset=['overall']).drop_duplicates(
            subset=['reviewerID', 'asin'], keep='last').reset_index(drop=True)
        cleaned_dataset = cleaned_dataset.sort_values(
            by=['reviewerID', 'unixReviewTime']).reset_index(drop=True)
        test_data_pre = cleaned_dataset[cleaned_dataset.overall >= 4.0].drop_duplicates(
            subset=['reviewerID'], keep='last')
        training_data = cleaned_dataset.drop(test_data_pre.index)
        test_data = test_data_pre[test_data_pre['reviewerID'].isin(
            training_data['reviewerID'])]
        return training_data, test_data

    def __clean_item_data(self):
        # this private function is used to clean meta data
        # sort the data among 'asin'
        # remove duplicate
        # keep the items inside test set and the training set
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

    def customer_ui_matrix(self, user_list=None, item_list=None, data_set=None):
        if type(user_list) == type(None):
            user_list = self.training_data['reviewerID'].drop_duplicates()
        if type(item_list) == type(None):
            item_list = self.training_data['asin'].drop_duplicates()
        if type(data_set) == type(None):
            data_set = self.training_data
        matrix = pd.DataFrame(columns=item_list, index=user_list)
        for row_i in range(len(data_set)):
            reviewerID = list(data_set.reviewerID)[row_i]
            asin = list(data_set.asin)[row_i]
            rate = list(self.training_data.overall)[row_i]
            if reviewerID in list(user_list) and asin in list(item_list):
                matrix.loc[reviewerID, asin] = rate
        return matrix


class Collaborativefiltering_recommendation(Preprocess):
    def __init__(self, ui_df=None, item_df=None, reuse: Preprocess = None):
        if type(reuse) != type(None):
            self.training_data = reuse.training_data
            self.test_data = reuse.test_data
            self.ui_matrix = reuse.ui_matrix
            self.clean_item_data = reuse.clean_item_data
        else:
            super().__init__(ui_df, item_df)
        reader = Reader(rating_scale=(1, 5))
        self.surprice_training = Dataset.load_from_df(
            self.training_data[['reviewerID', 'asin', 'overall']], reader=reader)

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

    def gridsearch(self, method: str, measures_method=['rmse'], cv_num=3, kwarg_dic: dict = None):
        training = self.surprice_training
        assert method == 'knn' or method == 'svd', 'method isn\'t knn or svd'
        if method == 'knn':
            if type(kwarg_dic) == type(None):
                param_grid = {'k': [1, 5, 10, 15],
                              'sim_options': {'name': ['msd', 'cosine', 'pearson', 'pearson_baseline'],
                                              # compute  similarities between items
                                              'user_based': [True]
                                              }
                              }
            else:
                param_grid = kwarg_dic
            gs = GridSearchCV(KNNWithMeans, param_grid,
                              measures=measures_method, cv=cv_num, n_jobs=-1)
        if method == 'svd':
            if type(kwarg_dic) == type(None):
                param_grid = {'n_epochs': [100, 200, 500, 800], 'n_factors': [
                    10, 30, 50], 'random_state': [0]}
            else:
                param_grid = kwarg_dic
            gs = GridSearchCV(
                SVD, param_grid, measures=measures_method, cv=cv_num, n_jobs=-1)
        gs.fit(training)
        return gs

    def knn(self, k=10, sim_options={'name': 'cosine', 'user_based': True}):
        training = self.surprice_training
        trainset = training.build_full_trainset()
        testset = trainset.build_anti_testset()
        model = KNNWithMeans(k=k, sim_options=sim_options)
        return model.fit(trainset).test(testset), model

    def svd(self, n_epochs=500, n_factors=30, random_state=0):
        training = self.surprice_training
        trainset = training.build_full_trainset()
        testset = trainset.build_anti_testset()
        model = SVD(n_epochs=n_epochs, n_factors=n_factors,
                    random_state=random_state)
        return model.fit(trainset).test(testset), model


class Evaluation(Preprocess):
    def __init__(self, prediction_list, ui_df=None, item_df=None, reuse: Preprocess = None):
        if type(reuse) != type(None):
            self.training_data = reuse.training_data
            self.test_data = reuse.test_data
            self.ui_matrix = reuse.ui_matrix
            self.clean_item_data = reuse.clean_item_data
        else:
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
                summation.append(copy.deepcopy(
                    self.ui_relevant_matrix.loc[user_id, item_id]))
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
                        filted_pred_list, user_id, i+1))
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
        for pred in self.clean_pred_list:
            user_item_rating[pred.uid].append((pred.iid, pred.est))
        for user_id, filted_pred_list in user_item_rating.items():
            filted_pred_list.sort(key=lambda x: x[1], reverse=True)
            summation.append(function_(filted_pred_list, user_id,
                             cut_off))
        return sum(summation)/float(num_users)

    def rank_k(self, clean: bool = False) -> dict:  # {uid:filted_pred_list}
        user_item_rating = defaultdict(list)
        if clean:
            pred_list = self.clean_pred_list
        else:
            pred_list = self.prediction_list
        for pred in pred_list:
            user_item_rating[pred.uid].append((pred.iid, pred.est))
        for user_id, filted_pred_list in user_item_rating.items():
            filted_pred_list.sort(key=lambda x: x[1], reverse=True)
        return user_item_rating

    def rmse(self):
        return surprise.accuracy.rmse(self.clean_pred_list)


class Contentbased_recommendation(Preprocess):

    def __init__(self, ui_df=None, item_df=None, word2vec: bool = False, bert: bool = False, reuse: Preprocess = None):
        if type(reuse) != type(None):
            self.training_data = reuse.training_data
            self.test_data = reuse.test_data
            self.ui_matrix = reuse.ui_matrix
            self.clean_item_data = reuse.clean_item_data
        else:
            super().__init__(ui_df, item_df)
        self.bert = bert
        self.original_title = list(self.clean_item_data['title'])
        if bert:
            self.bert_input, self.bert_lastlayer = self.__bert_encoding()
        else:
            self.clean_title = self.__clean_title()
        if word2vec:
            self.tfidf = self.__tfidf_by_word2vec()
        elif bert:
            self.tfidf = self.__tfidf_by_bert()
        else:
            self.tfidf = self.__tfidf()
        self.user_mean_vector = self.__user_mean_vector()

    def __clean_title(self):
        porter_stemmer = PorterStemmer()
        title_list = []
        for title in self.original_title:
            filter_list = [porter_stemmer.stem(word.lower()) for word in word_tokenize(
                title) if word.lower() not in stopwords.words("english") and word.isalpha()]
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
            temp_list = []
            for iid in item_in:
                temp_list.append(self.tfidf.loc[iid, :].to_numpy(
                )*int(self.ui_matrix.loc[uid, iid]))
            user_mean_vector_dic[uid] = np.sum(
                temp_list, axis=0)/len(self.training_data[self.training_data['reviewerID'] == uid]['asin'])
        return user_mean_vector_dic

    def print_words_num(self):
        words_list = []
        for title in self.clean_title:
            words_list += [word for word in word_tokenize(title)]
        print(len(set(words_list)))

    def pick_item_from_tfidf(self, item_list: list):
        vector_list = []
        for item in item_list:
            vector_list.append(self.tfidf.loc[item, :].to_numpy())
        return vector_list

    def predict(self):
        def x(bools): return True if bools == False else False
        prediction_list = []
        # print(user_list)
        # chick_1 = 0
        for uid in set(self.training_data['reviewerID']):
            item_not_in = self.clean_item_data['asin'][[x(bools) for bools in self.clean_item_data['asin'].isin(
                self.training_data[self.training_data['reviewerID'] == uid]['asin'])]]
            # chick_1 += len(item_not_in)
            vector_not_in = self.pick_item_from_tfidf(item_not_in)
            cos_sim = cosine_similarity(
                vector_not_in, [self.user_mean_vector[uid]])
            for i, item in enumerate(item_not_in):
                prediction_list.append(self.Prediction(
                    uid=uid, iid=item, est=cos_sim[i]))
        # print(chick_1)
        # print(len(prediction_list))
        return prediction_list

    def __tfidf_by_word2vec(self):
        word2vec_vectors = gensim.downloader.load('word2vec-google-news-300')
        sentences = [list(word_tokenize(title)) for title in self.clean_title]
        vector_list = []
        for sentence in sentences:
            vector = np.zeros((300,))
            mask = 0
            for word in sentence:
                try:
                    vector += word2vec_vectors[word]
                except KeyError:
                    continue
                else:
                    mask += 1
                    continue
            if mask == 0:
                mask += 1
            vector_list.append(vector/mask)
        return pd.DataFrame(np.array(vector_list), index=self.clean_item_data['asin'])

    def __bert_encoding(self):
        modelname = 'bert-base-uncased'
        tokenizer = BertTokenizerFast.from_pretrained(modelname)
        model = BertModel.from_pretrained(modelname).to(DEVICE)
        inputs = tokenizer(self.original_title, padding=True,
                           return_tensors="pt", return_attention_mask=True)
        outputs = model(**inputs)
        last_hidden_states = outputs[0]
        return inputs, last_hidden_states

    def __tfidf_by_bert(self):
        assert self.bert, 'disable bert model'
        mask_list = []
        vector_list = []
        return_list = []
        for i, item in enumerate(self.clean_item_data['asin']):
            mask_list.append(self.bert_input["attention_mask"][i])
            vector_list.append(self.bert_lastlayer[i])
            return_list.append(self.bert_input["attention_mask"][i].detach().numpy().dot(
                self.bert_lastlayer[i].detach().numpy())/np.sum(self.bert_input["attention_mask"][i].detach().numpy()))
        return pd.DataFrame(np.array(return_list), index=self.clean_item_data['asin'])


class Hybridrecommender_system(Preprocess):
    def __init__(self, evaluation_list: list[Evaluation], ui_df=None, item_df=None,
                 reuse: Preprocess = None, userprofile: dict = None, with_pred: str = None,
                 scalar: int = None, cut_off: int = -1):
        if type(reuse) != type(None):
            self.training_data = reuse.training_data
            self.test_data = reuse.test_data
            self.ui_matrix = reuse.ui_matrix
            self.clean_item_data = reuse.clean_item_data
        else:
            super().__init__(ui_df, item_df)
        self.evaluation_list = evaluation_list
        self.predictiondict_list = [eva.rank_k() for eva in evaluation_list]
        self.clean_predictiondict_list = [
            eva.rank_k(clean=True) for eva in evaluation_list]
        self.model_list = tuple(['model_{:d}'.format(
            i) for i in range(len(self. evaluation_list))])
        self.hybrid_dict = self.__form_pred_dic()
        if type(userprofile) != type(None):
            self.userprofile_dict = userprofile
        else:
            self.userprofile_dict = None
        if type(with_pred) != type(None):
            self.scalar = scalar
            self.ui_matrix = self.__fill_matrix(with_pred)
        self.__cut_off = cut_off

    def __form_pred_dic(self) -> dict:
        assert all([self.predictiondict_list[i].keys() == self.predictiondict_list[i+1].keys()
                   for i in range(len(self.predictiondict_list)-1)]), 'different user ID'
        key_list = self.predictiondict_list[0].keys()
        for key in key_list:
            temp_df_list = []
            for prediction_dict in self.predictiondict_list:
                if self.__cut_off != -1:
                    temp_df_list.append(pd.DataFrame.from_dict(
                        dict(prediction_dict[key][:self.__cut_off]), orient='index'))
                else:
                    temp_df_list.append(pd.DataFrame.from_dict(
                        dict(prediction_dict[key]), orient='index'))
            # print(key)
            # print(temp_df.head(10))
            yield key, temp_df_list

    def __fill_matrix(self, fill_with):
        ui_matrix = copy.deepcopy(self.ui_matrix)
        for key, item in self.predictiondict_list[fill_with].items():
            temp_df = pd.DataFrame(dict(item), index=[key])
            temp_df.loc[key, :] *= ((temp_df.loc[key, :]-temp_df.loc[key, :].min())/(
                temp_df.loc[key, :].max()-temp_df.loc[key, :].min()))*self.scalar
            ui_matrix.update(temp_df)
        return ui_matrix

    def reset_generator(self):
        self.hybrid_dict = self.__form_pred_dic()

    def weight_strategy(self, weight_list: list = [1, 1], borda: bool = False, rank: int = 5):

        def pd_norm(df_list: list[pd.DataFrame]):
            dflist = copy.deepcopy(df_list)
            for df in dflist:
                # if df[model].std() == 0:
                #     std_value = 1
                # else:
                #     std_value = df[model].std()
                df[0] = (df[0]-df[0].mean())/df[0].std()
            return dflist

        def weight_sum(df_list: list[pd.DataFrame], weight_list: list):
            weight_score = {}

            for df, weight in zip(df_list, weight_list):
                for row in df.iterrows():
                    if row[0] in weight_score:
                        weight_score[row[0]] += row[1][0]*weight
                    else:
                        weight_score[row[0]] = row[1][0]*weight
            return weight_score

        def borda_fuse(df_list: pd.DataFrame, rank: int = rank):
            all_rank_list = []
            bf_score = {}
            score = range(rank-1, -1, -1)
            for df in df_list:
                all_rank_list.append(
                    list(df.sort_values(by=[0], ascending=False).index))
            for i in range(rank):
                for rank_list in all_rank_list:
                    if rank_list[i] in bf_score:
                        bf_score[rank_list[i]] += score[i]
                    else:
                        bf_score[rank_list[i]] = score[i]
            return bf_score

        prediction_list = []
        if borda:
            for key, temp_df in self.hybrid_dict:
                user_temp_score = borda_fuse(temp_df)
                for iid, score in user_temp_score.items():
                    prediction_list.append(
                        self.Prediction(uid=key, iid=iid, est=score))
        else:
            for key, temp_df in self.hybrid_dict:
                user_temp_dp = pd_norm(temp_df)
                user_temp_score = weight_sum(user_temp_dp, weight_list)
                for iid, score in user_temp_score.items():
                    prediction_list.append(
                        self.Prediction(uid=key, iid=iid, est=score))
        self.reset_generator()
        return prediction_list

    def switching_strategy(self):

        prediction_list = []
        key_list = self.clean_predictiondict_list[0].keys()
        # iters = 0
        for key in key_list:
            max_evaluation = None
            max_rr = 0
            for i, evaluation in enumerate(self.evaluation_list):
                rr_score = evaluation.rr_k_user(self.clean_predictiondict_list[i][key], key, len(
                    self.clean_predictiondict_list[i][key]))
                # if iters %99==0:
                #     print(rr_score)
                # iters += 1
                if rr_score >= max_rr:
                    max_rr = rr_score
                    max_evaluation = i
            prediction_dict_uid = self.clean_predictiondict_list[max_evaluation][key]
            for item in prediction_dict_uid:
                prediction_list.append(self.Prediction(
                    uid=key, iid=item[0], est=item[1]))

        return prediction_list

    def metalevel_strategy(self, rank: int = 5):
        assert type(self.userprofile_dict) != type(
            None), 'disable meta level strategy, no user profile'

        def cos_sim_indf(userprofile_dict: dict, uid: str):
            uid_vector = userprofile_dict[uid]
            remain_uid_list = []
            remain_vector = []
            for id in userprofile_dict.keys():
                if id != uid:
                    remain_uid_list.append(id)
                    remain_vector.append(userprofile_dict[id])
            return remain_uid_list, cosine_similarity(remain_vector, [uid_vector])

        def x(bools): return True if bools == False else False
        prediction_list = []

        for uid in self.userprofile_dict.keys():
            remain_uid_list, cossim_list = cos_sim_indf(
                self.userprofile_dict, uid)
            top_index = np.argsort([-float(i) for i in cossim_list])[:rank]
            for iid in self.ui_matrix.columns[[x(bools) for bools in self.ui_matrix.columns.isin(
                    self.training_data[self.training_data['reviewerID'] == uid]['asin'])]]:
                prediction = 0
                summation = 0
                for index in top_index:
                    rank_uid = remain_uid_list[index]
                    rank_cossim = cossim_list[index]
                    prediction += self.ui_matrix.loc[rank_uid,
                                                     iid]*float(rank_cossim)
                    summation += float(rank_cossim)
                prediction /= summation
                prediction_list.append(self.Prediction(
                    uid=uid, iid=iid, est=prediction))

        return prediction_list


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
