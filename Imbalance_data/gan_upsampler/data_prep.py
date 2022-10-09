import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection

class Data_Prep():

    '''
    the pre-processing of input data and post-processing of generated data
    '''
    def __init__(self,raw_df,categorial,log,mixed,integer,types,test_ratio):

        self.categorical_cols = categorial 
        self.log_cols = log # list of skewed exponential numerical columns
        self.mixed_cols = mixed # list of mixed type columns ---> BUT actually columns with nan
        self.integer_cols = integer # list of numeric columns without floating number

        self.col_types = dict()
        self.col_types['categorical'] = list()
        self.col_types['mixed'] = dict()

        self.lower_bounds = dict()
        self.label_encoder_list = list()


        target_col = list(types.values())[0]
        
        y_real = raw_df[target_col]
        X_real = raw_df.drop(columns=[target_col])

        X_train_real,_,y_train_real,_ = model_selection.train_test_split(X_real,y_real,test_size=test_ratio,stratify=y_real,random_state=100)
        X_train_real[target_col] = y_train_real


        self.df = X_train_real
        # self.df.replace(r'',np.nan,inplace=True)
        # self.df.fillna('-99',inplace=True)

        # all_cols = set(self.df.columns)
        # irrelevant_missing_cols = set(self.categorical_cols)
        # relevant_missing_cols = list(all_cols - irrelevant_missing_cols)

        # for i in relevant_missing_cols:
        #     if i in list(self.mixed_cols.keys()):
        #         self.df[i] = self.df[i].apply(lambda x: -99 if x=='-99' else x)
        
        # log transformer for skewed exp numeric dist
        if self.log_cols:
            for log_col in self.log_cols:
                eps = 1
                lower = np.min(self.df.loc[self.df[log_col]!=-99][log_col].values)
                self.lower_bounds[log_col] = lower
                if lower > 0:
                    self.df[log_col] = self.df[log_col].apply(lambda x:np.log(x) if x!= -99 else -99)
                elif lower == 0:
                    self.df[log_col] = self.df[log_col].apply(lambda x:np.log(x+eps) if x!=-99 else -99)
                else:
                    self.df[log_col] = self.df[log_col].apply(lambda x:np.log(x-lower+eps) if x!=-99 else -99)
        


        for idx,col in enumerate(self.df.columns):

            if col in self.categorical_cols:
                label_encoder = preprocessing.LabelEncoder()
                self.df[col] = self.df[col].astype(str)
                label_encoder.fit(self.df[col])

                curr_label_encoder = dict()
                curr_label_encoder['col'] = col
                curr_label_encoder['label_encoder'] = label_encoder

                transformed_col = label_encoder.transform(self.df[col])
                self.df[col] = transformed_col

                self.label_encoder_list.append(curr_label_encoder)
                self.col_types['categorical'].append(idx)
            
            elif col in self.mixed_cols:
                self.col_types['mixed'][idx] = self.mixed_cols[col]
        
        super().__init__()

    
    def inverse_prep(self,data,eps=1):

        df_sample = pd.DataFrame(data,columns=self.df.columns)

        for i in range(len(self.label_encoder_list)):
            le = self.label_encoder_list[i]['label_encoder']
            df_sample[self.label_encoder_list[i]['col']] = df_sample[self.label_encoder_list[i]['col']].astype(int)
            df_sample[self.label_encoder_list[i]['col']] = le.inverse_transform(df_sample[self.label_encoder_list[i]['col']])

        
        if self.log_cols:
            for i in df_sample:
                if i in self.log_cols:
                    lower_bound = self.lower_bounds[i]
                    if lower_bound > 0:
                        df_sample[i] = df_sample[i].apply(lambda x:np.exp(x) if x!=-99 else -99)
                    elif lower_bound == 0:
                        df_sample[i] = df_sample[i].apply(lambda x: np.ceil(np.exp(x)-eps) if ((x!=-99) & ((np.exp(x)-eps) < 0)) else (np.exp(x)-eps if x!=-99 else -99))
                    else: 
                        df_sample[i] = df_sample[i].apply(lambda x: np.exp(x)-eps+lower_bound if x!=-99 else -99)
        
        if self.integer_cols:
            for col in self.integer_cols:
                df_sample[col] = (np.round(df_sample[col].values))
                df_sample[col] = df_sample[col].astype(int)
        
        return df_sample






