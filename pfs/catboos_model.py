from catboost import CatBoostRegressor
import shap
import pandas as pd
from io import BytesIO
import base64
from matplotlib import pyplot as plt



class SalesPrediction:
    def __init__(self, model_path):
        self.model = CatBoostRegressor().load_model(model_path)
        self.items_data = pd.read_csv('pfs/data/items.csv')
        self.shops_data = pd.read_csv('pfs/data/shop_df.csv')
        self.featues = ['date_block_num', 'shop_id', 'item_id', 'lag', 'item_category_id', 'shop_city', 'shop_location']
    

    def extract_features(self, request: dict):
        features = pd.DataFrame(columns=self.featues)

        features['date_block_num'] = [34]
        features['shop_id'] = [request['shop_id']]
        features['item_id'] = [request['item_id']]
        features['lag'] = [34 % 12]
        features['item_category_id'] = self.items_data[self.items_data['item_id'] == int(request['item_id'])]['item_category_id'].item()
        features['shop_city'] = self.shops_data[self.shops_data['shop_id'] == int(request['shop_id'])]['shop_city'].item()
        features['shop_location'] = self.shops_data[self.shops_data['shop_id'] == int(request['shop_id'])]['shop_location'].item()
        
        return features


    def predict(self, request: dict):
        features = self.extract_features(request)
        prediction = self.model.predict(features)
        return prediction
        
