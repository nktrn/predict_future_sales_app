from catboost import CatBoostRegressor


class SalesPrediction:
    def __init__(self, model_path):
        self.model = CatBoostRegressor().load_model(model_path)
    
    def predict(self, item_id):
        date_block_num = 34
        shop_id = 36
        item_category_id = 0
        shop_location = 'Москва'
        shop_city = 'ТРЦ'
        pred = self.model.predict([item_id, date_block_num, shop_id, item_category_id, shop_location, shop_city])
        return pred
