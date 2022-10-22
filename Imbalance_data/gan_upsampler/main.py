import pandas as pd
import time
from faker import Faker
from data_prep import Data_Prep
import warnings
warnings.filterwarnings('ignore')


class Let_us_fake():

    def __init__(self,
                 data_path,
                 test_ratio,
                 categorical_columns,
                 log_columns,
                 mixed_columns,
                 integer_columns,
                 problem_type,
                 epochs,
                 use_nn_mode):
        
        self.faker = Faker(epochs=epochs,use_nn_mode=use_nn_mode)
        self.raw_df = pd.read_csv(data_path)
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type

        self.epochs = epochs
        self.use_nn_mode = use_nn_mode
    
    def fit(self):

        start_time = time.time()
        self.data_prep = Data_Prep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.integer_columns,self.problem_type,self.test_ratio)
        print('data preparation done.')


        print('start training...')
        self.faker.fit(train_data=self.data_prep.df,
                       categorial=self.data_prep.col_types['categorical'],
                       mixed = self.data_prep.col_types['mixed'],
                       ml_task=self.problem_type)
        
        print('finish training in {} seconds.'.format(time.time() - start_time))


    def generate_samples(self):

        sample = self.faker.sample(len(self.raw_df))
        sample_df = self.data_prep.inverse_prep(sample)

        return sample_df



     