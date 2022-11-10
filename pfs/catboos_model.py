from catboost import CatBoostRegressor
import shap
import pandas as pd
from io import BytesIO
import base64
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt



class SalesPrediction:
    def __init__(self, model_path):
        self.model = CatBoostRegressor().load_model(model_path)
        self.items_data = pd.read_csv('pfs/data/items.csv')
        self.shops_data = pd.read_csv('pfs/data/shop_df.csv')
        self.featues = ['date_block_num', 'shop_id', 'item_id', 'lag', 'item_category_id', 'shop_city', 'shop_location']
        self.explainer = shap.TreeExplainer(self.model)
    

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
        shap_values = self.explainer(features)

        plt.figure(figsize=(10, 6), dpi=80)


        labels = range(len(shap_values.data[0]))
        labels_ = [
            f'{self.featues[i]}: {shap_values.data[0][i]}'
            for i in range(len(self.featues))
        ]
        plt.barh(labels, shap_values.values[0], height=1, color='r')
        plt.yticks(labels, labels_)


        buf = BytesIO()
        plt.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        img =  f"<img src='data:image/png;base64,{data}'/>"
        return prediction, img
        
