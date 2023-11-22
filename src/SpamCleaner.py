import pandas as pd
import os
import pandas as pd
class SpamCleaner():

    def __init__(self, df:pd.DataFrame) -> None:
        self._df = df
    
    def clean(self):
        self._normalize_data()
        return self._df

    def _normalize_data(self):
        "Normaliza las columnas que sean normalizables"
        for col in self._df.columns:
            self._df[col] = pd.to_numeric(self._df[col])
            min_in_col = self._df[col].min()
            max_in_col = self._df[col].max()
            normalized_serie = self._df[col].\
                map(lambda x: (x-min_in_col)/(max_in_col-min_in_col))
            self._df[col] = normalized_serie

    def split_df(self):
        "Divide el dataframe es entrenamiento y prueba con proporcion 80%-20%"
        spam = self._df[self._df[57]==1]
        no_spam = self._df[self._df[57]==0]
        spam_train = spam.head(1100)
        spam_test = spam.tail(len(spam)-len(spam_train))

        no_spam_train = no_spam.head(2480)
        no_spam_test = no_spam.tail(len(no_spam)-len(no_spam_train))

        train_data = pd.concat([spam_train, no_spam_train])
        test_data = pd.concat([spam_test, no_spam_test])

        return train_data, test_data
        
        


if __name__ == '__main__':
    df = pd.read_csv("spambase\\spambase.data", header=None)
    cleaner = SpamCleaner(df)
    cleaned_df = cleaner.clean()
    splitted_dfs = cleaner.split_df()
    splitted_dfs[0].to_csv("spam_train.csv", index=False)
    splitted_dfs[1].to_csv("spam_test.csv", index=False)