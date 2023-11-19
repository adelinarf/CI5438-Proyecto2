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

    def split_train(self):
        '''El dataset cuenta con 150 registros, donde cada especie tiene
        50 filas. Se toman 10 registros de cada especie para realizar
        las pruebas.
        
        Retorna dos df, uno con los datos de entrenamiento y otro con
        los datos de prueba'''        
        splitted_species_labeled = self.split_species()

        # Filtrar los df para tener solo 10 elementos.
        splitted_species = map(lambda data: data[1].head(10), splitted_species_labeled)
        test_data = pd.concat(splitted_species)

        # Eliminar del df de entrenamiento los datos de las pruebas
        train_data = self._df[~self._df.isin(test_data)].dropna()
        return train_data, test_data        
    
    def split_train_per_species(self, extra_data=5):
        '''Para cada especie, separa su respectivo dataset en dos
        archivos, uno de prueba y otro de entrenamiento. Con una proporcion
        de 80-20
        El archivo de prueba contiene datos de especies que NO corresponden
        a las de la especie para probar casos extra.

        Retorna una lista de tuplas de tamaño 3. Donde el primer elemento
        de la tupla es el nombre de la especie. El segundo son los datos de 
        prueba y el tercero los datos de entrenamiento
        '''
        splited_species = self.split_species()

        species_data = map(
            lambda data : (
                data[0], # Nombre de la especie
                data[1].head(10), # Datos de prueba                
                data[1][~self._df.isin(test_data)].dropna()), # Datos de entrenamiento
                splited_species)
        
        # Agregar los datos extras a cada df de prueba
        species_data = map(
            lambda data : (
                data[0], # Nombre de la especie
                pd.concat(
                    [data[1], self._df[self._df['species']!=data[0]].head(extra_data)]
                    ), # Datos de prueba                
                data[2]), # Datos de entrenamiento
                species_data)
        return species_data

    def split_species(self):
        '''Divide al dataset en tres datasets. Cada uno de estos
        datasets contienen todas los registros de una determinada
        especie de iris

        Retorna una lista de pares, donde el primer elemento contiene
            el nombre de la especie y el segundo el df
        '''
        species = self._df['species'].unique()
        
        splitted_species = [
            (specie, self._df[self._df['species']==specie]) for specie in species]
        return splitted_species

        
if __name__ == '__main__':
    
    df = pd.read_csv('iris.csv')
    cleaned_data = irisCleaner(df).clean()
    splitter = irisSplitter(cleaned_data)

    # Crear los datos de entrenamiento para el clasificador multiclase
    train_data, test_data = splitter.split_train()

    if not os.path.exists('iris data'):
        os.mkdir('iris data')
    
    train_data.to_csv(os.path.join('iris data', 'iris train data.csv'), index=False)
    test_data.to_csv(os.path.join('iris data', 'iris test data.csv'), index=False)

    # Crear los datos de entrenamiento para los clasificadores binarios
    splited_species = splitter.split_train_per_species()
    for label, test_df, train_df in splited_species:
        test_df.to_csv(os.path.join('iris data', f'only_{label}_test.csv'), index=False)
        train_df.to_csv(os.path.join('iris data', f'only_{label}_train.csv'), index=False)



    