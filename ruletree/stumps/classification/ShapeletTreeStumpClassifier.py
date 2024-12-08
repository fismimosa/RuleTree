import inspect
import io
import warnings
import tempfile
import numpy as np
from numba import UnsupportedError

from matplotlib import pyplot as plt

from ruletree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from ruletree.utils.data_utils import gini, entropy, get_info_gain
from ruletree.utils.define import DATA_TYPE_TS
from ruletree.utils.shapelet_transform.Shapelets import Shapelets


class ShapeletTreeStumpClassifier(DecisionTreeStumpClassifier):
    def __init__(self, n_shapelets=100,
                 n_shapelets_for_selection=np.inf, #int, inf, or 'stratified'
                 n_ts_for_selection_per_class=np.inf, #int, inf
                 sliding_window=50,
                 selection='random', #random, mi_clf, mi_reg, cluster
                 distance='euclidean',
                 mi_n_neighbors = 100,
                 random_state=42, n_jobs=1,
                 **kwargs):
        self.n_shapelets = n_shapelets
        self.n_shapelets_for_selection = n_shapelets_for_selection
        self.n_ts_for_selection_per_class = n_ts_for_selection_per_class
        self.sliding_window = sliding_window
        self.selection = selection
        self.distance = distance
        self.mi_n_neighbors = mi_n_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs

        if "max_depth" in kwargs and kwargs["max_depth"] > 1:
            warnings.warn("max_depth must be 1")

        kwargs["max_depth"] = 1

        super().__init__(**kwargs)

        kwargs |= {
            "n_shapelets": n_shapelets,
            "n_shapelets_for_selection": n_shapelets_for_selection,
            "n_ts_for_selection_per_class": n_ts_for_selection_per_class,
            "sliding_window": sliding_window,
            "selection": selection,
            "distance": distance,
            "mi_n_neighbors": mi_n_neighbors,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }

    def fit(self, X, y, sample_weight=None, check_input=True):
        if sample_weight is not None:
            raise UnsupportedError(f"sample_weight is not supported for {self.__class__.__name__}")

        self.st = Shapelets(n_shapelets=self.n_shapelets,
                            n_shapelets_for_selection=self.n_shapelets_for_selection,
                            n_ts_for_selection_per_class=self.n_ts_for_selection_per_class,
                            sliding_window=self.sliding_window,
                            selection=self.selection,
                            distance=self.distance,
                            mi_n_neighbors=self.mi_n_neighbors,
                            random_state=self.random_state,
                            n_jobs=self.n_jobs
                            )

        return super().fit(self.st.fit_transform(X), y, sample_weight=sample_weight, check_input=check_input)

    def apply(self, X, check_input=False):
        return super().apply(self.st.transform(X), check_input=check_input)

    def supports(self, data_type):
        return data_type in [DATA_TYPE_TS]

    def get_rule(self, columns_names=None, scaler=None, float_precision: int | None = 3):
        rule = {
            "feature_idx": self.feature_original[0],
            "threshold": self.threshold_original[0],
            "is_categorical": self.is_categorical,
            "samples": self.n_node_samples[0]
        }

        rule["feature_name"] = f"Shapelet_{rule['feature_idx']}"

        if scaler is not None:
            raise UnsupportedError(f"Scaler not supported for {self.__class__.__name__}")

        comparison = "<="
        not_comparison = ">"
        rounded_value = str(rule["threshold"]) if float_precision is None else round(rule["threshold"], float_precision)

        shape = self.st.shapelets[self.feature_original[0], 0]

        temp_dir = tempfile.TemporaryDirectory(prefix="RuleTree_", delete=False)
        with tempfile.TemporaryFile(dir=temp_dir.name,
                                    delete_on_close=False, delete=False,
                                    suffix=".png",
                                    mode="wb") as temp_file:
            plt.figure(figsize=(2, 1))
            plt.plot([i for i in range(shape.shape[0])], shape)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(temp_file, format="png", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(temp_file.name)

        rule["textual_rule"] = f"{rule["feature_name"]} {comparison} {rounded_value}\t{rule['samples']}"
        rule["blob_rule"] = f"{rule["feature_name"]} {comparison} {rounded_value}"
        rule["graphviz_rule"] = {
            "image": f'{temp_file.name}',
            "imagescale": "true",
            "imagepos": "bc",
            "label": f"{rule["feature_name"]} {comparison} {rounded_value}",
            "labelloc": "t"
        }

        rule["not_textual_rule"] = f"{rule["feature_name"]} {not_comparison} {rounded_value}"
        rule["not_blob_rule"] = f"{rule["feature_name"]} {not_comparison} {rounded_value}"
        rule["not_graphviz_rule"] = {
            "image": f'{temp_file.name}',
            "imagescale": "true",
            "label": f"{rule["feature_name"]} {not_comparison} {rounded_value}",
            "imagepos": "bc",
            "labelloc": "t"
        }

        return rule

    def node_to_dict(self):
        raise NotImplementedError()

    def dict_to_node(self, node_dict):
        raise NotImplementedError()