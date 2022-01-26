from abc import ABCMeta, abstractmethod
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import joblib
import math
import pandas as pd

class ITransfrom(metaclass=ABCMeta):

    @abstractmethod
    def transform(self):
        pass

    
class OneHotTransform(ITransfrom):

    def __init__(self, df:pd.DataFrame, save_enc:bool = False, enc_path:str = None, mod_hash_feature:list = None):
        self.df = df
        self.data = df.copy()
        self.save_enc = save_enc
        self.mod_hash_feature = mod_hash_feature

        self.one_hot_enc = None
        self.enc_path = enc_path
        if self.enc_path is None:
            self.one_hot_enc = OneHotEncoder(handle_unknown='ignore')
        else:
            self.one_hot_enc = self.load_encoder(path=self.enc_path)

    def transform(self)-> pd.DataFrame:
        if self.mod_hash_feature is not None:
        
            self.data[self.mod_hash_feature] = self.data[self.mod_hash_feature].applymap(lambda x : x%10)

        if self.enc_path is None:
            one_hot_data = self.one_hot_enc.fit_transform(self.data)
        else:
            one_hot_data = self.one_hot_enc.transform(self.data)


        if self.save_enc is True:
           joblib.dump(self.one_hot_enc, './storage/encoder/onehotencoder.save')  

        return pd.DataFrame(one_hot_data.toarray())


    @staticmethod
    def load_encoder(path:str = './storage/encoder/onehotencoder.save'):
        enc = joblib.load(path)
        return enc 



class NumericTransform(ITransfrom):
    def __init__(self, df:pd.DataFrame, want_scalar:bool, save_scaler:bool, scaler_path:str = None):
        """Doing the MinMax
        """
        self.data = df.copy()
        self.save_scaler = save_scaler

        self.scaler_path = scaler_path

        self.minmax = None
        if want_scalar:
            if self.scaler_path is None :
                self.minmax = MinMaxScaler()
            else:
                self.minmax = self.load_scaler(path=self.scaler_path)
            
    def transform(self) -> pd.DataFrame:
        self.data['loctm_hh'] = self.data['loctm'].apply(lambda x: math.floor(x/10000))
        self.data['loctm_mm'] = self.data['loctm'].apply(lambda x: math.floor(x/100)-math.floor(x/10000)*100)
        self.data['loctm_ss'] = self.data['loctm'].apply(lambda x: math.floor(x)-math.floor(x/100)*100)


        if self.minmax is not None:
            if self.scaler_path is None:
                numeric_data_minmax = self.minmax.fit_transform(self.data)
            else:
                numeric_data_minmax = self.minmax.transform(self.data)

            res = pd.DataFrame(numeric_data_minmax, columns=self.data.columns)

        else:
            res = self.data
            

        if self.save_scaler:
            joblib.dump(self.minmax, './storage/scaler/MinMax.save') 

        return res

    
    @staticmethod
    def load_scaler(path:str = './storage/scaler/MinMax.save'):
        scaler = joblib.load(path)
        return scaler 

    



