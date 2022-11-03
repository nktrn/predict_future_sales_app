from flask import Flask
from neptune.new import init_model
from pfs.constans import *
from pfs.catboos_model import SalesPrediction

app = Flask(__name__)

nmodel = init_model(
    project=neptune_project,
    api_token=neptune_api_token,
    with_id=model_id
)
nmodel['model/signature'].download(model_name)

model = SalesPrediction(model_name)


from pfs import forecast 
