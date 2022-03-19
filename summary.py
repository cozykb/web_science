import gzip
import os
import json
import re
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from surprise import Reader
from surprise import Dataset

class Preprocess:
    def __init__(self,ui_df:pd.DataFrame,item_df:pd.DataFrame):
        self.ui_df = ui_df
        self.item_df = item_df

    class Predection:
        def __init__(self, uid, iid, est):  
            self.uid = uid
            self.iid = iid
            self.est = est

    def _split_ui_data(self):
        df = self.ui_df.sort_values(by=['reviewerID', 'asin', 'unixReviewTime'])
        cleaned_dataset = df.dropna(subset=['overall']).drop_duplicates(subset=['reviewerID', 'asin'], keep = 'last').reset_index(drop=True)
        # print(len(cleaned_dataset))
        # cleaned_dataset.head()
        cleaned_dataset = cleaned_dataset.sort_values(by=['reviewerID', 'unixReviewTime']).reset_index(drop=True)
        # extracting the latest (in time) positively rated item (rating  ≥4 ) by each user. 
        test_data_pre = cleaned_dataset[cleaned_dataset.overall >= 4.0].drop_duplicates(subset=['reviewerID'], keep='last')
        # generate training data
        training_data = cleaned_dataset.drop(test_data_pre.index)
        # Remove users that do not appear in the training set.
        user_in_training = test_data_pre['reviewerID'].isin(training_data['reviewerID'])
        test_data = test_data_pre[user_in_training]
        return training_data, test_data

    training_data = _split_ui_data[0]
    test_data = _split_ui_data[1]

    def _clean_item_data(self):
        df = self.item_df
        # Discard duplicates
        df = df.sort_values(by=['asin'])
        clean_dataset_item = df.drop_duplicates(subset=['asin'], keep = 'last').reset_index(drop=True)
        # Discard items that weren't rated by our subset of users
        item_in_subset = list(Preprocess.test_data.loc[:,'asin'])+list(Preprocess.training_data.loc[:,'asin'])
        # print(list(item_in_subset))
        clean_dataset_item = clean_dataset_item[clean_dataset_item['asin'].isin(item_in_subset)]
        return clean_dataset_item

    clean_item_data = _clean_item_data

    def _ui_matrix_training(self):
        matrix = pd.DataFrame(columns = Preprocess.training_data['asin'].drop_duplicates(), index = Preprocess.training_data['reviewerID'].drop_duplicates())
        for row_i in range(len(Preprocess.training_data)):
            reviewerID = list(Preprocess.training_data.reviewerID)[row_i]
            asin = list(Preprocess.training_data.asin)[row_i]
            rate = list(Preprocess.training_data.overall)[row_i]
            matrix.loc[reviewerID, asin] = rate
        matrix = matrix.fillna(0)
        return matrix
    
    ui_matrix = _ui_matrix_training

    # def _ui_relevant_matrix(self):
    #     relevant_matrix = pd.DataFrame(columns = Preprocess.test_data['asin'].drop_duplicates(), index = Preprocess.test_data['reviewerID'].drop_duplicates())
    #     for row_i in range(len(Preprocess.test_data)):
    #         reviewerID = list(Preprocess.test_data.reviewerID)[row_i]
    #         asin = list(Preprocess.test_data.asin)[row_i]
    #         rate = list(Preprocess.test_data.overall)[row_i]
    #         relevant_matrix.loc[reviewerID, asin] = 1 if rate >= 4 else 0
    #     relevant_matrix = relevant_matrix.fillna(0)
    #     return relevant_matrix
    
    # ui_relevant_matrix = _ui_relevant_matrix



class Collaborativefiltering_recommendation(Preprocess):

    def cos_sim_ui(user_id:str, item_id:str):
        cosine_similarities = cosine_similarity(np.array(Preprocess.ui_matrix.loc[Preprocess.ui_matrix.index == user_id]),Preprocess.ui_matrix[Preprocess.ui_matrix.loc[:,item_id]>0])
        result = pd.DataFrame(Preprocess.ui_matrix[Preprocess.ui_matrix.loc[:,item_id]>0].loc[:,item_id])
        result.insert(result.shape[1],'cos_sim',np.array(cosine_similarities).T)
        return result

    def cos_sim_prediction(user_id:str, item_id:str, rank:int):
        result = Collaborativefiltering_recommendation.cos_sim_ui(user_id, item_id)
        result_sort = result.sort_values(by='cos_sim', ascending=False).head(rank)
        prediction = 0
        for index in result_sort.index:
            prediction += result_sort.loc[index][0]*result_sort.loc[index][1]
        prediction /= result_sort.loc[:,'cos_sim'].sum()
        return prediction

    def simple_svd(user_id:str, item_id:str, k:int, mute:bool):
        matrix = pd.DataFrame(columns = Preprocess.training_data['asin'].drop_duplicates(), index = Preprocess.training_data['reviewerID'].drop_duplicates())
        for row_i in range(len(Preprocess.training_data)):
            reviewerID = list(Preprocess.training_data.reviewerID)[row_i]
            asin = list(Preprocess.training_data.asin)[row_i]
            rate = list(Preprocess.training_data.overall)[row_i]
            matrix.loc[reviewerID, asin] = rate
        matrix = matrix.fillna(0)
        for index in matrix.index:
            mean_ = np.mean([i for i  in matrix.loc[index] if i != 0])
            if index==user_id:
                user_id_mean = mean_
            for item in range(len(matrix.loc[index])):
                if matrix.loc[index][item] != 0:
                    matrix.loc[index][item] -= mean_
        Q, sigma, P = svds(matrix, k)
        U = np.dot(Q, np.diag(sigma))
        user_factors = pd.DataFrame(data = U, index = matrix.index)
        item_factors = pd.DataFrame(data = P, columns = matrix.columns)
        if not mute:
            print(user_factors.loc[user_id])
            print(item_factors.loc[:,item_id])
        prediction = user_id_mean + np.dot(np.array(user_factors.loc[user_id]),np.array(item_factors.loc[:,item_id]))
        return prediction



class Contentbased_recommendation(Preprocess):
    def __init__(self):
        pass
    class Predection:
        def __init__(self, uid, iid, est):  
            self.uid = uid
            self.iid = iid
            self.est = est
    

    