from catboost import CatBoostRegressor
import shap
import pandas as pd
from io import BytesIO
import base64
from matplotlib import pyplot as plt


class SalesPrediction:
    def __init__(self, model_path):
        self.model = CatBoostRegressor().load_model(model_path)
    
    def predict(self, shop_id):
        data = [34, int(shop_id['shop_id']), 30, 10, 40, 'Балашиха', 'shop. center']
        predict = self.model.predict(data)
        f = ['date_block_num', 'shop_id', 'item_id', 'lag', 'item_category_id', 'shop_city', 'shop_location']
        label = ['label']
        df = pd.DataFrame(
            columns=f + label
        )
        data += [predict]

        df.loc[len(df)] = data

        explainer = shap.TreeExplainer(self.model)

        shap_values = explainer(df[f])


        buf = BytesIO()
        #shap.force_plot(explainer.expected_value, shap_values, df[f], matplotlib = True, show = False)
        shap.waterfall_plot(shap_values[0], show = False)

        plt.savefig(buf,
            format = "png",
            dpi = 150,
            bbox_inches = 'tight')
        
        dataToTake = base64.b64encode(buf.getbuffer()).decode("ascii")
        return dataToTake, predict
        
