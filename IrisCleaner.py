import pandas as pd
import os
class irisCleaner():

    def __init__(self, df:pd.DataFrame) -> None:
        self._df = df
    
    def clean(self):
        self._normalize_data()
        return self._df

    def _normalize_data(self):
        "Normaliza las columnas que sean normalizables"
        colums_to_normalize = [
            'sepal_length','sepal_width','petal_length','petal_width'
        ]

        for col in colums_to_normalize:
            self._df[col] = pd.to_numeric(self._df[col])
            min_in_col = self._df[col].min()
            max_in_col = self._df[col].max()
            normalized_serie = self._df[col].\
                map(lambda x: (x-min_in_col)/(max_in_col-min_in_col))
            self._df[col] = normalized_serie

class irisSplitter():
    def __init__(self, df:pd.DataFrame) -> None:
        self._df = df

    def split(self):
        # El dataset cuenta con 150 registros, donde cada especie tiene
        # 50 filas. Se toman 10 registros de cada especie para realizar
        # las pruebas
        species = self._df['species'].unique()
        splitted_species = [self._df[self._df['species']==specie].head(10) for specie in species]        
        test_data = pd.concat(splitted_species)
        train_data = self._df[~self._df.isin(test_data)].dropna()
        return train_data, test_data        
        
if __name__ == '__main__':
    df = pd.read_csv('iris.csv')
    cleaned_data = irisCleaner(df).clean()
    splitter = irisSplitter(cleaned_data)
    train_data, test_data = splitter.split()

    if not os.path.exists('iris data'):
        os.mkdir('iris data')

    train_data.to_csv(os.path.join('iris data', 'iris train data.csv'), index=False)
    test_data.to_csv(os.path.join('iris data', 'iris test data.csv'), index=False)

    