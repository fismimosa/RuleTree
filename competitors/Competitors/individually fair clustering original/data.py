import pandas as pd

age_dict = {'[0-10)': 1, '[10-20)': 2, '[20-30)': 3, '[30-40)': 4,
            '[40-50)': 5, '[50-60)': 6, '[60-70)': 7, '[70-80)': 8, '[80-90)': 9, '[90-100)': 10}
age_map = lambda x: age_dict[x]

gender_dict = {'Male': 1, 'Female': 2, 'Unknown/Invalid': 0}


def load_data(dataset_name, feature_set):
    distance_feature_dict = {}
    fairness_feature_dict = {}
    if dataset_name == 'diabetes':
        discrete_distance_features = feature_set['discrete_distance']
        discrete_fairness_features = feature_set['discrete_fairness']
        continuous_distance_features = feature_set['continuous_distance']
        continuous_fairness_features = feature_set['continuous_fairness']
        distance_features = discrete_distance_features + continuous_distance_features
        fairness_features = discrete_fairness_features + continuous_fairness_features
        discrete_features = discrete_distance_features + discrete_fairness_features
        continuous_features = continuous_distance_features + continuous_fairness_features
        all_features = distance_features + fairness_features

        df_diab = pd.read_csv('./Data/dataset_diabetes/diabetic_data.csv', delimiter=",", usecols=['patient_nbr'] + all_features)

        for discrete_feature in discrete_features:
            df_diab[discrete_feature] = df_diab[discrete_feature].apply(str)

        for continuous_feature in continuous_features:
            if continuous_feature == 'age':
                df_diab['age'] = df_diab['age'].apply(age_map)
            df_diab[continuous_feature] = df_diab[continuous_feature].apply(float)
            df_diab[continuous_feature] = (df_diab[continuous_feature] - df_diab[continuous_feature].min()) / (df_diab[continuous_feature].max() - df_diab[continuous_feature].min())

        for discrete_feature in discrete_features:
            df_diab = df_diab[df_diab[discrete_feature] != '?']

        for discrete_feature in discrete_features:
            x = pd.get_dummies(df_diab[discrete_feature], prefix=discrete_feature)
            df_diab = pd.concat([x, df_diab], axis=1)
            df_diab = df_diab.drop([discrete_feature], axis=1)

        all_distance_cols = []
        for distance_feature in distance_features:
            all_cols = [col_name for col_name in df_diab.columns if distance_feature in col_name]
            all_distance_cols += all_cols

        all_fairness_cols = []
        for fairness_feature in fairness_features:
            all_cols = [col_name for col_name in df_diab.columns if fairness_feature in col_name]
            all_fairness_cols += all_cols

        df_distance = df_diab[['patient_nbr'] + all_distance_cols]
        df_fairness = df_diab[['patient_nbr'] + all_fairness_cols]

        distance_feature_dict = df_distance.set_index('patient_nbr').T.to_dict('list')
        fairness_feature_dict = df_fairness.set_index('patient_nbr').T.to_dict('list')
    elif dataset_name == 'adult':
        discrete_distance_features = feature_set['discrete_distance']
        discrete_fairness_features = feature_set['discrete_fairness']
        continuous_distance_features = feature_set['continuous_distance']
        continuous_fairness_features = feature_set['continuous_fairness']
        distance_features = discrete_distance_features + continuous_distance_features
        fairness_features = discrete_fairness_features + continuous_fairness_features
        discrete_features = discrete_distance_features + discrete_fairness_features
        continuous_features = continuous_distance_features + continuous_fairness_features
        all_features = distance_features + fairness_features

        df_adult = pd.read_csv('./Data/adult.data', delimiter=", ", engine='python')
        df_adult.columns = ['age', 'workclass', 'fnlwgt', 'education', 'educationnum', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capitalloss', 'hoursperweek', 'nativecountry', 'salary']
        df_adult = df_adult[all_features]
        df_adult['index'] = df_adult.index
        # df_adult = df_adult.reset_index(level=0)

        for discrete_feature in discrete_features:
            df_adult[discrete_feature] = df_adult[discrete_feature].apply(str)

        for continuous_feature in continuous_features:
            # if continuous_feature == 'age':
            #     df_adult['age'] = df_adult['age'].apply(age_map)
            df_adult[continuous_feature] = df_adult[continuous_feature].apply(float)
            df_adult[continuous_feature] = (df_adult[continuous_feature] - df_adult[continuous_feature].min()) / (df_adult[continuous_feature].max() - df_adult[continuous_feature].min())

        for discrete_feature in discrete_features:
            df_adult = df_adult[df_adult[discrete_feature] != '?']

        for discrete_feature in discrete_features:
            x = pd.get_dummies(df_adult[discrete_feature], prefix=discrete_feature)
            df_adult = pd.concat([x, df_adult], axis=1)
            df_adult = df_adult.drop([discrete_feature], axis=1)

        all_distance_cols = []
        for distance_feature in distance_features:
            all_cols = [col_name for col_name in df_adult.columns if distance_feature in col_name]
            all_distance_cols += all_cols

        all_fairness_cols = []
        for fairness_feature in fairness_features:
            all_cols = [col_name for col_name in df_adult.columns if fairness_feature in col_name]
            all_fairness_cols += all_cols

        df_distance = df_adult[['index'] + all_distance_cols]
        df_fairness = df_adult[['index'] + all_fairness_cols]

        distance_feature_dict = df_distance.set_index('index').T.to_dict('list')
        fairness_feature_dict = df_fairness.set_index('index').T.to_dict('list')
    elif dataset_name == 'bank':
        discrete_distance_features = feature_set['discrete_distance']
        discrete_fairness_features = feature_set['discrete_fairness']
        continuous_distance_features = feature_set['continuous_distance']
        continuous_fairness_features = feature_set['continuous_fairness']
        distance_features = discrete_distance_features + continuous_distance_features
        fairness_features = discrete_fairness_features + continuous_fairness_features
        discrete_features = discrete_distance_features + discrete_fairness_features
        continuous_features = continuous_distance_features + continuous_fairness_features
        all_features = distance_features + fairness_features

        df_bank = pd.read_csv('./Data/bank/bank.csv', delimiter=";")
        df_bank = df_bank[all_features]
        df_bank['index'] = df_bank.index
        # df_adult = df_adult.reset_index(level=0)

        for discrete_feature in discrete_features:
            df_bank[discrete_feature] = df_bank[discrete_feature].apply(str)

        for continuous_feature in continuous_features:
            # if continuous_feature == 'age':
            #     df_adult['age'] = df_adult['age'].apply(age_map)
            df_bank[continuous_feature] = df_bank[continuous_feature].apply(float)
            df_bank[continuous_feature] = (df_bank[continuous_feature] - df_bank[continuous_feature].min()) / (df_bank[continuous_feature].max() - df_bank[continuous_feature].min())

        for discrete_feature in discrete_features:
            df_bank = df_bank[df_bank[discrete_feature] != '?']
            df_bank = df_bank[df_bank[discrete_feature] != 'unknown']

        for discrete_feature in discrete_features:
            x = pd.get_dummies(df_bank[discrete_feature], prefix=discrete_feature)
            df_bank = pd.concat([x, df_bank], axis=1)
            df_bank = df_bank.drop([discrete_feature], axis=1)

        all_distance_cols = []
        for distance_feature in distance_features:
            all_cols = [col_name for col_name in df_bank.columns if distance_feature in col_name]
            all_distance_cols += all_cols

        all_fairness_cols = []
        for fairness_feature in fairness_features:
            all_cols = [col_name for col_name in df_bank.columns if fairness_feature in col_name]
            all_fairness_cols += all_cols

        df_distance = df_bank[['index'] + all_distance_cols]
        df_fairness = df_bank[['index'] + all_fairness_cols]

        distance_feature_dict = df_distance.set_index('index').T.to_dict('list')
        fairness_feature_dict = df_fairness.set_index('index').T.to_dict('list')

    return distance_feature_dict, fairness_feature_dict


def select_feature_set_diabetes():
    discrete_distance_features = []
    continuous_distance_features = ['age', 'number_emergency']
    discrete_fairness_features = []
    continuous_fairness_features = ['time_in_hospital', 'num_lab_procedures']

    feature_set = {
        'discrete_distance': discrete_distance_features,
        'discrete_fairness': discrete_fairness_features,
        'continuous_distance': continuous_distance_features,
        'continuous_fairness': continuous_fairness_features
    }
    return feature_set


def select_feature_set_adult():
    discrete_distance_features = []
    continuous_distance_features = ['age', 'educationnum']
    discrete_fairness_features = ['salary']
    continuous_fairness_features = ['hoursperweek']

    feature_set = {
        'discrete_distance': discrete_distance_features,
        'discrete_fairness': discrete_fairness_features,
        'continuous_distance': continuous_distance_features,
        'continuous_fairness': continuous_fairness_features
    }
    return feature_set


def select_feature_set_bank():
    discrete_distance_features = []
    continuous_distance_features = ['duration', 'age']
    discrete_fairness_features = ['education']
    continuous_fairness_features = ['balance']

    feature_set = {
        'discrete_distance': discrete_distance_features,
        'discrete_fairness': discrete_fairness_features,
        'continuous_distance': continuous_distance_features,
        'continuous_fairness': continuous_fairness_features
    }
    return feature_set
