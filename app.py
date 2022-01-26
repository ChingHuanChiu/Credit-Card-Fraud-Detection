from typing import Optional, List, Dict


import logging
import tensorflow as tf
import numpy as np
import pandas as pd
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, Query, Form
from pydantic import BaseModel
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


from data.transform import OneHotTransform, NumericTransform
from data.sql_orm import insert_todb
# from model.supervised import MultiInputCredictFraudDetect


app = FastAPI(version='0.0.0.',
              title='Credict Card Detection')

templates = Jinja2Templates(directory="template/")

print("Finishing Loading...")




def get_logger(filename, level, format, log_name):
   
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    formater = logging.Formatter(format)
    handler = logging.FileHandler(filename, encoding="utf8")
    handler.setFormatter(formater)
    logger.addHandler(handler)

    return logger

error_logger = get_logger('./log/error.log', logging.INFO, format="%(asctime)s : %(message)s", log_name='error')
info_logger = get_logger('./log/api.log', logging.INFO, format="%(asctime)s : %(message)s", log_name='info')



# @app.exception_handler(RequestValidationError)
# async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
#     error_logger.info(f"ERROR :{request.method} {request.url}  {exc.errors()}")
#     return JSONResponse({"code": "400", "message": exc.errors()})



@app.get("/")
async def form_post(request: Request):
    return templates.TemplateResponse('index.html', context={'request': request, 'result': ''})


MODEL = None

@app.post("/")
async def form_post(request: Request, 
                    bacno=Form(...),
                    locdt=Form(...),
                    loctm=Form(...),
                    cano=Form(...),
                    contp=Form(...),
                    etymd=Form(...),
                    mchno=Form(...),
                    acqic=Form(...),
                    mcc=Form(...),
                    conam=Form(...),
                    ecfg=Form(...),
                    insfg=Form(...),
                    iterm=Form(...),
                    stocn=Form(...),
                    scity=Form(...),
                    stscd=Form(...),
                    ovrlt=Form(...),
                    flbmk=Form(...),
                    hcefg=Form(...),
                    csmcu=Form(...),
                    flg_3dsmk=Form(...),
                    txkey=Form(...)
                    

):
    data = {'acqic': acqic , 'bacno': bacno, 'cano': cano, 'conam':conam, 'contp': contp, 'csmcu': csmcu, 'ecfg': ecfg, 'etymd': etymd, 
            'flbmk': flbmk, 'flg_3dsmk': flg_3dsmk, 'hcefg': hcefg, 'insfg': insfg, 'iterm': iterm, 'locdt': locdt, 'loctm': loctm, 
            'mcc': mcc, 'mchno': mchno, 'ovrlt': ovrlt, 'scity': scity, 'stocn': stocn, 'stscd': stscd, 'txkey': txkey}


    try:
        # TODO: Using Basemodel

        info_logger.info(f"INFO :{request.method} {request.url} {data}")

        input_data = [int(acqic), int(bacno), int(cano), float(conam), contp, csmcu, ecfg, etymd, flbmk, flg_3dsmk, 
                    hcefg, insfg, float(iterm), float(locdt), float(loctm), int(mcc), int(mchno), ovrlt, int(scity), 
                    int(stocn), stscd, int(txkey)]

        

        data_df = pd.DataFrame([input_data], columns=data.keys()) 

        NUMERIC_FEATURE = ["locdt", "loctm", "conam", "iterm"]
        ONE_HOT_FEATURE = ["contp", "etymd", "ecfg", "insfg", "stscd", "ovrlt", "flbmk", "hcefg", "flg_3dsmk", "csmcu"]
        MOD_HASH_ONE_HOT_FEATURE = ["scity", "stocn", "bacno", "txkey", "cano", "mchno", "acqic", "mcc"]

        
        one_hot_train_data = OneHotTransform(data_df[ONE_HOT_FEATURE + MOD_HASH_ONE_HOT_FEATURE], 
                                        save_enc=False,
                                        enc_path='./storage/encoder/onehotencoder.save',
                                        mod_hash_feature=MOD_HASH_ONE_HOT_FEATURE).transform()



        
        numeric_df = NumericTransform(data_df[NUMERIC_FEATURE], want_scalar=True, save_scaler=False, scaler_path='./storage/scaler/MinMax.save').transform()

        # one_hot_input = one_hot_train_data
        # numeric_input = numeric_df
        input_data = pd.concat([one_hot_train_data, numeric_df], 1)
        global MODEL
        if MODEL is None:

            MODEL = tf.keras.models.load_model('./storage/model/supervised/autoencoder')


        res, confidence = predict(MODEL, input_data)


        result = {"result":res, "confidence":confidence}

        data.update({'predict': result['result']})
        insert_todb(data, error_logger)


        return templates.TemplateResponse('index.html', context={'request': request, 'result': result})
    
    except Exception as e:
        error_logger.info(f"ERROR :{request.method} {request.url}  {e} {data}") 
        return JSONResponse({"code": "400", "message": "make sure the right data!"})



              
def predict(model, input):
    with tf.device('/cpu:0'):
        d = {1: '盜刷', 0: '正常'}

        # output = model(one_hot_input, numeric_df)
        output = model.predict(input)

        res = d[np.argmax(output, -1)[0]]
        print(output)
        confidence = tf.math.reduce_max(output).numpy()
        return res, str(confidence)



