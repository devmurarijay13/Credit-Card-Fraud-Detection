import os
import sys
import pandas as pd
from dataclasses import dataclass

from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def train_and_evaluate(self,train_arr,test_arr):
        try:
            logging.info('spliting train and test data')

            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            neg, pos = (y_train==0).sum(),(y_train==1).sum()

            ## already tested model in notebook
            model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=neg/pos,
                eval_metric='logloss',
                random_state=42
            )

            model.fit(X_train,y_train)
            logging.info('model is trained')

            # y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            threshold = 0.7
            y_pred = (y_prob >= threshold).astype(int)


            print("Precision Score :",precision_score(y_test,y_pred))
            print('Recall Score :',recall_score(y_test,y_pred))
            print("f1 Score :",f1_score(y_test,y_pred))

            print("Confusion Matrix : \n")
            print(confusion_matrix(y_test,y_pred))

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=model
            )
        
        except Exception as e:
            raise CustomException(e,sys)

