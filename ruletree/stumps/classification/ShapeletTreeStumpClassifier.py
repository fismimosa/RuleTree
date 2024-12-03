import inspect

from pyts.transformation import ShapeletTransform
from sklearn.tree import DecisionTreeClassifier

from ruletree.base.RuleTreeBaseStump import RuleTreeBaseStump
from ruletree.utils.data_utils import gini, entropy, get_info_gain
from ruletree.utils.define import DATA_TYPE_TS


class ShapeletTreeStumpClassifier(DecisionTreeClassifier, RuleTreeBaseStump):
    def __init__(self, **kwargs):
        super().__init__({k: v for k, v in kwargs.items() if k in inspect.getfullargspec(super().__init__)[0]})

        self.st = ShapeletTransform(
            {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(ShapeletTransform.__init__)[0]}
        )

        self.kwargs = kwargs
        self.unique_val_enum = None

        self.threshold_original = None
        self.feature_original = None
        self.coefficients = None

        if 'criterion' not in kwargs or kwargs['criterion'] == "gini":
            self.impurity_fun = gini
        elif kwargs['criterion'] == "entropy":
            self.impurity_fun = entropy
        else:
            self.impurity_fun = kwargs['criterion']

    def get_params(self, deep=True):
        return self.kwargs

    def fit(self, X, y, sample_weight=None, check_input=True):
        best_info_gain = -float('inf')

        if len(self.numerical) > 0:
            super().fit(X[:, self.numerical], y, sample_weight=sample_weight, check_input=check_input)
            self.feature_original = self.tree_.feature
            self.threshold_original = self.tree_.threshold
            self.n_node_samples = self.tree_.n_node_samples
            best_info_gain = get_info_gain(self)

        self._fit_cat(X, y, best_info_gain)

        return self


    def supports(data_type):
        return data_type in [DATA_TYPE_TS]

    def get_rule(self, columns_names=None, scaler=None, float_precision: int | None = 3):
        raise NotImplementedError()

    def node_to_dict(self):
        raise NotImplementedError()

    def dict_to_node(self, node_dict):
        raise NotImplementedError()