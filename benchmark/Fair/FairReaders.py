import pandas as pd
from sklearn.preprocessing import LabelEncoder


def readTitanic():
    df = pd.read_csv('../../datasets/CLF/titanic.csv').drop(columns=['PassengerId', 'Embarked', 'Cabin_letter'])
    df = df[~df.Cabin_n.isin(['T', 'D'])]

    df['Sex'] = LabelEncoder().fit_transform(df.Sex)

    columns = set(df.columns.tolist()) - {'Survived', 'Sex'}

    return df[['Sex']+list(columns)+['Survived']].rename(columns={'Survived':'y'}).astype("float")