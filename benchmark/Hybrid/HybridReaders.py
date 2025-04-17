import pandas as pd
from pandas.core.dtypes.common import is_object_dtype
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def read_titanic(encode=True):
    sensible_col = 'Sex'
    target_col = 'Survived'
    df = pd.read_csv('../../datasets/CLF/titanic.csv').drop(columns=['Embarked', 'PassengerId'])

    if encode:
        df.Sex = LabelEncoder().fit_transform(df.Sex)
        df.Cabin_n = LabelEncoder().fit_transform(df.Cabin_n)

        ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
                               remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=12)

        df = pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out())

    columns = set(df.columns.tolist()) - {sensible_col, target_col}

    return 'titanic', df[[sensible_col] + list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype(
        "float")


def read_bank(encode=True):
    target_col = 'give_credit'
    df = pd.read_csv("../../datasets/CLF/bank.csv")

    columns = set(df.columns.tolist()) - {target_col}

    return 'bank', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_diabetes(encode=True):
    target_col = 'Outcome'
    df = pd.read_csv("../../datasets/CLF/diabetes.csv")

    columns = set(df.columns.tolist()) - {target_col}

    return ('diabetes', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float"))


def read_taiwan_credit(encode=True):
    target_col = 'dpnm'
    df = pd.read_csv("../../datasets/CLU/FAIR/taiwan_credit_sensible.csv").sort_values(by='AGE_BINNED')

    if encode:
        df.AGE_BINNED = LabelEncoder().fit_transform(df.AGE_BINNED)

    if encode:
        columns_to_increment = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        df[columns_to_increment] = df[columns_to_increment] + 2

    columns = set(df.columns.tolist()) - {target_col}

    return 'taiwan_credit', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_compass(encode=True):
    target_col = 'two_year_recid'
    df = pd.read_csv("../../datasets/CLU/compas-scores-two-years.csv", index_col=0)
    df.drop(columns=['id', 'compas_screening_date', 'dob', 'decile_score', 'c_jail_in',
                     'c_jail_out', 'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'c_charge_degree',
                     'c_charge_desc', 'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc',
                     'r_jail_in', 'r_jail_out', 'vr_charge_degree', 'type_of_assessment', 'decile_score.1',
                     'screening_date', 'v_type_of_assessment', 'v_decile_score', 'v_screening_date',
                     'in_custody', 'out_custody', 'start', 'end', 'event'], inplace=True)

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

    columns = set(df.columns.tolist()) - {target_col}

    return 'compass', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_adult(encode=True):
    target_col = 'class'
    df = pd.read_csv("../../datasets/CLF/adult.csv")
    df = df.drop(columns=['fnlwgt', 'education'])

    if encode:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])
        df.sex = LabelEncoder().fit_transform(df.sex)

        ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
                               remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0,
                               n_jobs=12)
        df = pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out())

    columns = set(df.columns.tolist()) - {target_col}

    return 'adult', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_auction(encode=True):
    target_col = 'verification.result'
    df = pd.read_csv("../../datasets/CLF/auction.csv")
    df = df.drop(columns=['verification.time'])

    columns = set(df.columns.tolist()) - {target_col}

    return 'auction', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_fico(encode=True):
    target_col = 'RiskPerformance'
    df = pd.read_csv("../../datasets/CLF/fico.csv")

    if encode:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    columns = set(df.columns.tolist()) - {target_col}

    return 'fico', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_german_credit(encode=True):
    target_col = 'default'
    df = pd.read_csv("../../datasets/CLF/german_credit.csv")
    df = df.drop(columns=['installment_as_income_perc', 'present_res_since', 'credits_this_bank',
                          'people_under_maintenance'])

    if encode:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])
        df['foreign_worker'] = LabelEncoder().fit_transform(df['foreign_worker'])
        ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
                               remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0,
                               n_jobs=12)
        df = pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out())

    columns = set(df.columns.tolist()) - {target_col}

    return 'german_credit', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_home(encode=True):
    target_col = 'in_sf'
    df = pd.read_csv("../../datasets/CLF/home.csv")

    columns = set(df.columns.tolist()) - {target_col}

    return 'home', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_ionosphere(encode=True):
    target_col = 'class'
    df = pd.read_csv("../../datasets/CLF/ionosphere.csv")

    if encode:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    columns = set(df.columns.tolist()) - {target_col}

    return 'ionosphere', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_iris(encode=True):
    target_col = 'class'
    df = pd.read_csv("../../datasets/CLF/iris.csv")

    if encode:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    columns = set(df.columns.tolist()) - {target_col}

    return 'iris', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_vehicle(encode=True):
    target_col = 'CLASS'
    df = pd.read_csv("../../datasets/CLF/vehicle.csv")

    if encode:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    columns = set(df.columns.tolist()) - {target_col}

    return 'vehicle', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_wdbc(encode=True):
    target_col = 'diagnosis'
    df = pd.read_csv("../../datasets/CLF/wdbc.csv")

    if encode:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    columns = set(df.columns.tolist()) - {target_col}

    return 'wdbc', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_wine(encode=True):
    target_col = 'quality'
    df = pd.read_csv("../../datasets/CLF/wine.csv")

    if encode:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    columns = set(df.columns.tolist()) - {target_col}

    return 'wine', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


small_datasets = [read_iris,
                  #read_auction,
                  #read_home,
                  #read_diabetes,
                  #read_titanic
                  ]


medium_datasets = [read_wine, read_compass, read_vehicle, read_wdbc, read_ionosphere]


big_datasets = [read_bank, read_taiwan_credit, read_fico, read_german_credit, read_adult]


all_datasets = [read_titanic, read_bank, read_diabetes, read_taiwan_credit, read_compass, read_adult, read_auction,
                read_fico, read_german_credit, read_home, read_ionosphere, read_iris, read_vehicle, read_wdbc,
                read_wine]

#assert len(all_datasets) == len(small_datasets) + len(medium_datasets) + len(big_datasets)

if __name__ == "__main__":
    dataset_shapes = []
    for dataset_reader in all_datasets:
        dataset_name, df = dataset_reader()
        dataset_shapes.append((dataset_name, df.shape[1], df.shape[0]))

    dataset_shapes.sort(key=lambda x: x[1])

    for dataset_name, num_columns, num_rows in dataset_shapes:
        print(f"{dataset_name}: ({num_rows}, {num_columns})")
