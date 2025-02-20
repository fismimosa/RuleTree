import pandas as pd
from pandas.core.dtypes.common import is_object_dtype
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def read_titanic(encode=True):
    sensible_col = 'Sex'
    target_col = 'Survived'
    df = pd.read_csv('../../datasets/CLU/FAIR/titanic.csv').drop(columns=['Embarked'])

    if encode:
        df['Sex'] = LabelEncoder().fit_transform(df.Sex)

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return 'titanic_sex', df[[sensible_col]+list(columns)+[target_col]].rename(columns={target_col:'y'}).astype("float")

def _read_bank(encode=True):
    df = pd.read_csv("../../datasets/CLU/FAIR/bank_full_sensible.csv").sort_values('AGE_BINNED')

    if encode:
        edu_map = {
            'unknown': 0,
            'primary': 1,
            'secondary': 2,
            'tertiary': 3,
        }

        df['AGE_BINNED'] = LabelEncoder().fit_transform(df.AGE_BINNED)
        df['loan'] = LabelEncoder().fit_transform(df.loan)
        df['housing'] = LabelEncoder().fit_transform(df.housing)
        df['default'] = LabelEncoder().fit_transform(df.default)
        df['y'] = LabelEncoder().fit_transform(df.y)
        df['education'] = df['education'].apply(lambda x: edu_map[x])

        ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
            remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=12)

        df = pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out())

    return df

def read_bank_age(encode=True):
    sensible_col = 'AGE_BINNED'
    target_col = 'y'
    df = _read_bank(encode)

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return ('bank_'+sensible_col, df[[sensible_col] + list(columns) + [target_col]].rename(columns={target_col: 'y'})
            .astype("float"))

def read_bank_default(encode=True):
    sensible_col = 'default'
    target_col = 'y'
    df = _read_bank(encode)

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return ('bank_'+sensible_col, df[[sensible_col] + list(columns) + [target_col]].rename(columns={target_col: 'y'})
            .astype("float"))

def read_bank_education(encode=True):
    sensible_col = 'education'
    target_col = 'y'
    df = _read_bank(encode)

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return ('bank_'+sensible_col, df[[sensible_col] + list(columns) + [target_col]].rename(columns={target_col: 'y'})
            .astype("float"))

def read_bank_housing(encode=True):
    sensible_col = 'housing'
    target_col = 'y'
    df = _read_bank(encode)

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return ('bank_'+sensible_col, df[[sensible_col] + list(columns) + [target_col]].rename(columns={target_col: 'y'})
            .astype("float"))

def read_bank_marital(encode=True):
    sensible_col = 'marital'
    target_col = 'y'
    df = pd.read_csv("../../datasets/CLU/FAIR/bank_full_sensible.csv").sort_values('AGE_BINNED')

    if encode:
        edu_map = {
            'unknown': 0,
            'primary': 1,
            'secondary': 2,
            'tertiary': 3,
        }

        df['AGE_BINNED'] = LabelEncoder().fit_transform(df.AGE_BINNED)
        df['loan'] = LabelEncoder().fit_transform(df.loan)
        df['housing'] = LabelEncoder().fit_transform(df.housing)
        df['default'] = LabelEncoder().fit_transform(df.default)
        df['y'] = LabelEncoder().fit_transform(df.y)
        df['education'] = df['education'].apply(lambda x: edu_map[x])
        df['marital'] = LabelEncoder().fit_transform(df.marital)

        ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
                               remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=12)

        df = pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out())

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return ('bank_'+sensible_col, df[[sensible_col] + list(columns) + [target_col]].rename(columns={target_col: 'y'})
            .astype("float"))

def _read_diabetes(encode=True):
    df = pd.read_csv("../../datasets/CLU/FAIR/diabetes_data_sensible.csv").sort_values(by='age')

    if encode:
        df.age = LabelEncoder().fit_transform(df.age)
        df.gender = LabelEncoder().fit_transform(df.gender)
        for col_names in list(df.columns):
            if not is_object_dtype(df[col_names]):
                continue

            if len(df[col_names].unique()) == 2:
                df[col_names] = LabelEncoder().fit_transform(df[col_names])
            if len(df[col_names].unique()) > 10 or len(df[col_names].unique()) <= 1:
                df = df.drop(columns=col_names)

        ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
                               remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0,
                               n_jobs=12)

        df = pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out())

    return df

def read_diabetes_age(encode=True):
    sensible_col = 'age'
    target_col = 'diabetesMed'
    df = _read_diabetes(encode)

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return ('diabetes_'+sensible_col, df[[sensible_col] + list(columns) + [target_col]].rename(columns={target_col: 'y'})
            .astype("float"))

def read_diabetes_gender(encode=True):
    sensible_col = 'gender'
    target_col = 'diabetesMed'
    df = _read_diabetes(encode)

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return ('diabetes_'+sensible_col, df[[sensible_col] + list(columns) + [target_col]].rename(columns={target_col: 'y'})
            .astype("float"))

def read_diabetes_race(encode=True):
    sensible_col = 'race'
    target_col = 'diabetesMed'
    df = pd.read_csv("../../datasets/CLU/FAIR/diabetes_data_sensible.csv").sort_values(by='age')

    if encode:
        df.age = LabelEncoder().fit_transform(df.age)
        df.race = LabelEncoder().fit_transform(df.race)

        for col_names in df.columns:
            if len(df[col_names].unique()) == 2:
                df[col_names] = LabelEncoder().fit_transform(df[col_names])

        ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
                               remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0,
                               n_jobs=12)

        df = pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out())

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return ('diabetes_' + sensible_col, df[[sensible_col] + list(columns) + [target_col]].rename(columns={target_col: 'y'})
            .astype("float"))

def _read_taiwan_credit(encode=True):
    df = pd.read_csv("../../datasets/CLU/FAIR/taiwan_credit_sensible.csv").sort_values(by='AGE_BINNED')

    if encode:
        df.AGE_BINNED = LabelEncoder().fit_transform(df.AGE_BINNED)

    return df

def read_taiwan_credit_age(encode=True):
    sensible_col = 'AGE_BINNED'
    target_col = 'dpnm'
    df = _read_taiwan_credit(encode)

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return (
    'taiwan_credit_' + sensible_col, df[[sensible_col] + list(columns) + [target_col]].rename(columns={target_col: 'y'})
    .astype("float"))

def read_taiwan_credit_education(encode=True):
    sensible_col = 'EDUCATION'
    target_col = 'dpnm'
    df = _read_taiwan_credit(encode)

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return (
    'taiwan_credit_' + sensible_col, df[[sensible_col] + list(columns) + [target_col]].rename(columns={target_col: 'y'})
    .astype("float"))

def read_taiwan_credit_marriage(encode=True):
    sensible_col = 'MARRIAGE'
    target_col = 'dpnm'
    df = _read_taiwan_credit(encode)

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return (
    'taiwan_credit_' + sensible_col, df[[sensible_col] + list(columns) + [target_col]].rename(columns={target_col: 'y'})
    .astype("float"))

def read_taiwan_credit_sex(encode=True):
    sensible_col = 'SEX'
    target_col = 'dpnm'
    df = _read_taiwan_credit(encode)

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return (
    'taiwan_credit_' + sensible_col, df[[sensible_col] + list(columns) + [target_col]].rename(columns={target_col: 'y'})
    .astype("float"))


def _read_compass(encode=True):
    df = pd.read_csv("../../datasets/CLU/compas-scores-two-years.csv", index_col=0)
    df.drop(columns=["id", "compas_screening_date", "age", "c_charge_desc", "r_charge_desc", "score_text", "start",
                     "end", ], inplace=True)

    if encode:
        df.race = LabelEncoder().fit_transform(df.race)
        df.age_cat = LabelEncoder().fit_transform(df.age_cat)
        df.v_score_text = LabelEncoder().fit_transform(df.v_score_text)

        for col_names in df.columns:
            if not is_object_dtype(df[col_names]):
                continue

            if len(df[col_names].unique()) == 2:
                df[col_names] = LabelEncoder().fit_transform(df[col_names])
            if len(df[col_names].unique()) > 10 or len(df[col_names].unique()) <= 1:
                df = df.drop(columns=col_names)

        ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
                               remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0,
                               n_jobs=12)

        df = pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out())

    return  df

def read_compass_sex(encode=True):
    sensible_col = 'sex'
    target_col = 'two_year_recid'
    df = _read_compass(encode)
    df.drop(columns=[x for x in df.columns if 'race' in x])

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return (
        'compass_' + sensible_col,
        df[[sensible_col] + list(columns) + [target_col]].rename(columns={target_col: 'y'})
        .astype("float"))

def read_compass_age(encode=True):
    sensible_col = 'age_cat'
    target_col = 'two_year_recid'
    df = _read_compass(encode)
    df.drop(columns=[x for x in df.columns if 'race' in x])

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return (
        'compass_' + sensible_col,
        df[[sensible_col] + list(columns) + [target_col]].rename(columns={target_col: 'y'})
        .astype("float"))

def read_compass_race(encode=True):
    sensible_col = 'race'
    target_col = 'two_year_recid'
    df = _read_compass(encode)

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return (
        'compass_' + sensible_col,
        df[[sensible_col] + list(columns) + [target_col]].rename(columns={target_col: 'y'})
        .astype("float"))

if __name__ == "__main__":
    """dataset_name, df = read_titanic()
    
    dataset_name, df = read_bank_age()
    dataset_name, df = read_bank_default()
    dataset_name, df = read_bank_education()
    dataset_name, df = read_bank_housing()
    dataset_name, df = read_bank_marital()"""

    #dataset_name, df = read_diabetes_age()
    #dataset_name, df = read_diabetes_gender()
    #dataset_name, df = read_diabetes_race()

    """dataset_name, df = read_taiwan_credit_age()
    dataset_name, df = read_taiwan_credit_education()
    dataset_name, df = read_taiwan_credit_marriage()
    dataset_name, df = read_taiwan_credit_sex()"""

    dataset_name, df = read_compass_age()
    dataset_name, df = read_compass_race()
    dataset_name, df = read_compass_sex()

    print(df)