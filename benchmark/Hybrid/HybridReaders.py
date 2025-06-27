import pandas as pd
from pandas.core.dtypes.common import is_object_dtype
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from ucimlrepo import fetch_ucirepo


def read_titanic(encode=True):
    target_col = 'Survived'
    df = pd.read_csv('../../../datasets/CLF/titanic.csv').drop(columns=['Embarked', 'PassengerId'])

    if encode:
        df.Sex = LabelEncoder().fit_transform(df.Sex)
        df.Cabin_n = LabelEncoder().fit_transform(df.Cabin_n)
        df.Cabin_letter = df.Cabin_letter.apply(lambda x: ord(x) - ord('A')) #lower = front

        #ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
        #                       remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0, n_jobs=12)

        #df = pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out())

    columns = set(df.columns.tolist()) - {target_col}

    return 'titanic', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_bank(encode=True):
    target_col = 'give_credit'
    df = pd.read_csv("../../../datasets/CLF/bank.csv")

    columns = set(df.columns.tolist()) - {target_col}

    return 'bank', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_diabetes(encode=True):
    target_col = 'Outcome'
    df = pd.read_csv("../../../datasets/CLF/diabetes.csv")

    columns = set(df.columns.tolist()) - {target_col}

    return ('diabetes', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float"))


def read_taiwan_credit(encode=True):
    target_col = 'dpnm'
    df = pd.read_csv("../../../datasets/CLU/FAIR/taiwan_credit_sensible.csv").sort_values(by='AGE_BINNED')

    if encode:
        df.AGE_BINNED = LabelEncoder().fit_transform(df.AGE_BINNED)

    if encode:
        columns_to_increment = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        df[columns_to_increment] = df[columns_to_increment] + 2

    columns = set(df.columns.tolist()) - {target_col}

    return 'taiwan_credit', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_compass(encode=True):
    target_col = 'two_year_recid'
    df = pd.read_csv("../../../datasets/CLU/compas-scores-two-years.csv", index_col=0)
    df.drop(columns=['id', 'compas_screening_date', 'dob', 'decile_score', 'c_jail_in',
                     'c_jail_out', 'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'c_charge_degree',
                     'c_charge_desc', 'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc',
                     'r_jail_in', 'r_jail_out', 'vr_charge_degree', 'type_of_assessment', 'decile_score.1',
                     'screening_date', 'v_type_of_assessment', 'v_decile_score', 'v_screening_date',
                     'in_custody', 'out_custody', 'start', 'end', 'event'], inplace=True)

    if encode:
        score_text = {
            'Low': 0,
            'Medium': 1,
            'High': 2,
        }

        df.race = LabelEncoder().fit_transform(df.race)
        df.age_cat = LabelEncoder().fit_transform(df.age_cat)
        df.v_score_text = LabelEncoder().fit_transform(df.v_score_text)
        df.score_text = df.score_text.apply(lambda x: score_text[x])

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
    df = pd.read_csv("../../../datasets/CLF/adult.csv")
    df = df.drop(columns=['fnlwgt', 'education', 'native-country'])
    df = df.map(lambda x: pd.NA if str(x).strip() == '?' else x).dropna()

    if encode:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])
        df.sex = LabelEncoder().fit_transform(df.sex)
        workclass = {
            'Never-worked': 0,
            'Without-pay': 1,
            'Self-emp-not-inc': 2,
            'Self-emp-inc': 3,
            'Private': 4,
            'Local-gov': 5,
            'State-gov': 6,
            'Federal-gov': 7,
            '?': -1  # or use 8 if you prefer to treat it as a separate valid category
        }
        df.workclass = df.workclass.apply(lambda x: workclass[x.strip()])

        relationship = {
            'Husband': 0,
            'Wife': 1,
            'Own-child': 2,
            'Other-relative': 3,
            'Not-in-family': 4,
            'Unmarried': 5
        }

        df.relationship = df.relationship.apply(lambda x: relationship[x.strip()])


        ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
                               remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0,
                               n_jobs=12)
        df = pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out())

    columns = set(df.columns.tolist()) - {target_col}

    return 'adult', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_auction(encode=True):
    target_col = 'verification.result'
    df = pd.read_csv("../../../datasets/CLF/auction.csv")
    df = df.drop(columns=['verification.time'])

    columns = set(df.columns.tolist()) - {target_col}

    return 'auction', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_fico(encode=True):
    target_col = 'RiskPerformance'
    df = pd.read_csv("../../../datasets/CLF/fico.csv")

    if encode:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    columns = set(df.columns.tolist()) - {target_col}

    return 'fico', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_german_credit(encode=True):
    target_col = 'default'
    df = pd.read_csv("../../../datasets/CLF/german_credit.csv")
    df = df.drop(columns=['installment_as_income_perc', 'present_res_since', 'credits_this_bank', 'telephone',
                          'people_under_maintenance', 'other_debtors', 'other_installment_plans'])

    df['sex'] = df.personal_status_sex.apply(lambda x: x.split(' ')[0])
    df['personal_status'] = df.personal_status_sex.apply(lambda x: x.split(' ')[-1])
    df.drop(columns=['personal_status_sex'], inplace=True)

    if encode:
        account_check_status = {
            '< 0 DM': 0,
            '0 <= ... < 200 DM': 1,
            '>= 200 DM / salary assignments for at least 1 year': 2,
            'no checking account': -1
        }

        savings = {
            '... < 100 DM': 0,
            '100 <= ... < 500 DM': 1,
            '500 <= ... < 1000 DM': 2,
            '.. >= 1000 DM': 3,
            'unknown/ no savings account': -1
        }

        present_emp_since = {
            'unemployed': 0,
            '... < 1 year': 1,
            '1 <= ... < 4 years': 2,
            '4 <= ... < 7 years': 3,
            '.. >= 7 years': 4
        }

        job = {
            'unemployed/ unskilled - non-resident': 0,
            'unskilled - resident': 1,
            'skilled employee / official': 2,
            'management/ self-employed/ highly qualified employee/ officer': 3,
        }

        housing = {
            'for free': 0,
            'rent': 1,
            'own': 2,
        }

        credit_history = {
            'no credits taken/ all credits paid back duly': 0,
            'all credits at this bank paid back duly': 1,
            'existing credits paid back duly till now': 2,
            'critical account/ other credits existing (not at this bank)': 3,
            'delay in paying off in the past': 4
        }

        personal_status = {
            'single': 0,
            'divorced/separated': 1,
            'divorced/separated/married': 2,
            'married/widowed': 3
        }

        df['account_check_status'] = df['account_check_status'].apply(lambda x: account_check_status[x.strip()])
        df['savings'] = df['savings'].apply(lambda x: savings[x.strip()])
        df['present_emp_since'] = df['present_emp_since'].apply(lambda x: present_emp_since[x.strip()])
        df['job'] = df['job'].apply(lambda x: job[x.strip()])
        df['housing'] = df['housing'].apply(lambda x: housing[x.strip()])
        df['credit_history'] = df['credit_history'].apply(lambda x: credit_history[x.strip()])
        df['personal_status'] = df['personal_status'].apply(lambda x: personal_status[x.strip()])

        df[target_col] = LabelEncoder().fit_transform(df[target_col])
        df['sex'] = LabelEncoder().fit_transform(df['sex'])
        df['foreign_worker'] = LabelEncoder().fit_transform(df['foreign_worker'])
        ct = ColumnTransformer([("cat", OneHotEncoder(), make_column_selector(dtype_include="object"))],
                               remainder='passthrough', verbose_feature_names_out=False, sparse_threshold=0,
                               n_jobs=12)
        df = pd.DataFrame(ct.fit_transform(df), columns=ct.get_feature_names_out())

    columns = set(df.columns.tolist()) - {target_col}

    return 'german_credit', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_home(encode=True):
    target_col = 'in_sf'
    df = pd.read_csv("../../../datasets/CLF/home.csv")

    columns = set(df.columns.tolist()) - {target_col}

    return 'home', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_ionosphere(encode=True):
    target_col = 'class'
    df = pd.read_csv("../../../datasets/CLF/ionosphere.csv")

    if encode:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    columns = set(df.columns.tolist()) - {target_col}

    return 'ionosphere', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_iris(encode=True):
    target_col = 'class'
    df = pd.read_csv("../../../datasets/CLF/iris.csv")

    if encode:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    columns = set(df.columns.tolist()) - {target_col}

    return 'iris', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_vehicle(encode=True):
    target_col = 'CLASS'
    df = pd.read_csv("../../../datasets/CLF/vehicle.csv")

    if encode:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    columns = set(df.columns.tolist()) - {target_col}

    return 'vehicle', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_wdbc(encode=True):
    target_col = 'diagnosis'
    df = pd.read_csv("../../../datasets/CLF/wdbc.csv")

    if encode:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    columns = set(df.columns.tolist()) - {target_col}

    return 'breast', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_wine(encode=True):
    target_col = 'quality'
    df = pd.read_csv("../../../datasets/CLF/wine.csv")

    if encode:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    columns = set(df.columns.tolist()) - {target_col}

    return 'wine', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")

def read_breast(encode=True):
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    X = breast_cancer_wisconsin_diagnostic.data.features.copy()
    y = breast_cancer_wisconsin_diagnostic.data.targets

    if encode:
        y = LabelEncoder().fit_transform(y.values.ravel())

    X['y'] = y

    return 'breast', X.astype("float")

def read_page(encode=True):
    page_blocks_classification = fetch_ucirepo(id=78)

    X = page_blocks_classification.data.features.copy()
    y = page_blocks_classification.data.targets

    X['y'] = y

    return 'page', X.astype("float")

def read_sonar(encode=True):
    connectionist_bench_sonar_mines_vs_rocks = fetch_ucirepo(id=151)

    X = connectionist_bench_sonar_mines_vs_rocks.data.features.copy()
    y = connectionist_bench_sonar_mines_vs_rocks.data.targets

    if encode:
        y = LabelEncoder().fit_transform(y.values.ravel())

    X['y'] = y

    return 'sonar', X.astype("float")


def read_vertebral(encode=True):
    vertebral_column = fetch_ucirepo(id=212)

    X = vertebral_column.data.features.copy()
    y = vertebral_column.data.targets

    if encode:
        y = LabelEncoder().fit_transform(y.values.ravel())

    X['y'] = y

    return 'vertebral', X.astype("float")


def read_heloc(encode=True):
    target_col = 'RiskPerformance'
    df = pd.read_csv("../../../datasets/CLF/heloc.csv")

    if encode:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    columns = set(df.columns.tolist()) - {target_col}

    return 'heloc', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")

def read_algerian(encode=True):
    target_col = 'label'
    df = pd.read_csv("../../../datasets/CLF/algerian_forest_fires.csv").drop(columns=['year'])

    columns = set(df.columns.tolist()) - {target_col}

    return 'algerian', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_seeds(encode=True):
    target_col = 'label'
    df = pd.read_csv("../../../datasets/CLF/seeds.csv")

    columns = set(df.columns.tolist()) - {target_col}

    return 'seeds', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")

def read_glass(encode=True):
    target_col = 'label'
    df = pd.read_csv("../../../datasets/CLF/glass.csv")

    columns = set(df.columns.tolist()) - {target_col}

    return 'glass', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")

def read_ecoli(encode=True):
    target_col = 'label'
    df = pd.read_csv("../../../datasets/CLF/ecoli.csv")

    columns = set(df.columns.tolist()) - {target_col}

    return 'ecoli', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_magic(encode=True):
    target_col = 'label'
    df = pd.read_csv("../../../datasets/CLF/ecoli.csv")

    columns = set(df.columns.tolist()) - {target_col}

    return 'magic', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")

def read_covertype(encode=True):
    target_col = 'label'
    df = pd.read_csv("../../../datasets/CLF/covertype.csv")

    columns = set(df.columns.tolist()) - {target_col}

    return 'covertype', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_eye(encode=True):
    target_col = 'label'
    df = pd.read_csv("../../../datasets/CLF/eye.csv")

    columns = set(df.columns.tolist()) - {target_col}

    return 'eye', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_sylvine(encode=True):
    target_col = 'label'
    df = pd.read_csv("../../../datasets/CLF/sylvine.csv")

    columns = set(df.columns.tolist()) - {target_col}

    return 'sylvine', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


def read_bankMarketing(encode=True):
    target_col = 'label'
    df = pd.read_csv("../../../datasets/CLF/bank-marketing.csv")

    columns = set(df.columns.tolist()) - {target_col}

    return 'bankMarketing', df[list(columns) + [target_col]].rename(columns={target_col: 'y'}).astype("float")


small_datasets = [
    read_iris,
    read_vertebral,
    read_home,
    read_auction,
    read_seeds,
    read_ecoli,
    read_bankMarketing,
    read_magic,
    read_diabetes,
    read_titanic,
]


medium_datasets = [
    read_glass,
    read_covertype,
    read_page,
    read_algerian,
    read_compass,
    read_wine,
    read_vehicle,
    read_ionosphere,
]


big_datasets = [
    read_breast,
    read_eye,
    read_sylvine,
    read_heloc,
    read_german_credit,
    read_taiwan_credit,
    read_adult,
    read_bank,
    read_sonar,
]


all_datasets = small_datasets + medium_datasets + big_datasets

if __name__ == "__main__":
    dataset_shapes = []
    for dataset_reader in all_datasets:
        dataset_name, df = dataset_reader()
        dataset_shapes.append((dataset_name, df.shape[1], df.shape[0]))

    dataset_shapes.sort(key=lambda x: x[1])

    for dataset_name, num_columns, num_rows in dataset_shapes:
        print(f"{dataset_name}: ({num_rows}, {num_columns})")
