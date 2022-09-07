# import table
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer


# data processor class
class Processor:

    def __init__(self, datatypes):
        self.datatypes = datatypes
    
    # fit encoding
    def fit(self, matrix):
        preprocessors, cutoffs = [], []
        for i, (column, datatype) in enumerate(self.datatypes):
            preprocessed_col = matrix[:,i].reshape(-1, 1)

            if 'categorical' in datatype:
                preprocessor = LabelBinarizer()
            else:
                preprocessor = MinMaxScaler()

            preprocessed_col = preprocessor.fit_transform(preprocessed_col)
            cutoffs.append(preprocessed_col.shape[1])
            preprocessors.append(preprocessor)
        
        self.cutoffs = cutoffs
        self.preprocessors = preprocessors
    
    # transform data
    def transform(self, matrix):
        preprocessed_cols = []
        
        for i, (column, datatype) in enumerate(self.datatypes):
            preprocessed_col = matrix[:,i].reshape(-1, 1)
            preprocessed_col = self.preprocessors[i].transform(preprocessed_col)
            preprocessed_cols.append(preprocessed_col)

        return np.concatenate(preprocessed_cols, axis=1)

    # fit and transform data
    def fit_transform(self, matrix):
        self.fit(matrix)
        return self.transform(matrix)

    # reverse data encoding  
    def inverse_transform(self, matrix):
        postprocessed_cols = []

        j = 0
        for i, (column, datatype) in enumerate(self.datatypes):
            postprocessed_col = self.preprocessors[i].inverse_transform(matrix[:,j:j+self.cutoffs[i]])

            if 'categorical' in datatype:
                postprocessed_col = postprocessed_col.reshape(-1, 1)
            else:
                if 'positive' in datatype:
                    postprocessed_col = postprocessed_col.clip(min=0)

                if 'int' in datatype:
                    postprocessed_col = postprocessed_col.round()

            postprocessed_cols.append(postprocessed_col)
            
            j += self.cutoffs[i]
        
        return np.concatenate(postprocessed_cols, axis=1)
