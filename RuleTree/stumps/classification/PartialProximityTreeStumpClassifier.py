import copy
import random
import warnings

import numpy as np
import psutil
import tempfile312
from matplotlib import pyplot as plt
from numba import UnsupportedError
from tqdm.auto import tqdm

from RuleTree.stumps.classification.DecisionTreeStumpClassifier import DecisionTreeStumpClassifier
from RuleTree.utils.define import DATA_TYPE_TS, DATA_TYPE_TABULAR
from RuleTree.utils.shapelet_transform.Shapelets import Shapelets
from RuleTree.utils.shapelet_transform.TabularShapelets import TabularShapelets


class PartialProximityTreeStumpClassifier(DecisionTreeStumpClassifier):
    def __init__(self,
                 n_shapelets=psutil.cpu_count(logical=False) * 2,
                 n_ts_for_selection=100,  # int, inf
                 n_features_strategy=2,
                 selection='random',  # random, mi_clf, mi_reg, cluster
                 distance='euclidean',
                 proximity_on_same_features=True,
                 scaler=None,
                 use_combination=True,
                 random_state=42, n_jobs=1,
                 **kwargs):
        self.n_shapelets = n_shapelets
        self.n_ts_for_selection = n_ts_for_selection
        self.n_features_strategy = n_features_strategy
        self.selection = selection
        self.distance = distance
        self.proximity_on_same_features = proximity_on_same_features
        self.scaler = scaler
        self.use_combination = use_combination
        self.random_state = random_state
        self.n_jobs = n_jobs

        if "max_depth" in kwargs and kwargs["max_depth"] > 1:
            warnings.warn("max_depth must be 1")

        kwargs["max_depth"] = 1

        if selection not in ["random", "cluster", "all"]:
            raise ValueError("'selection' must be 'random', 'all' or 'cluster'")

        super().__init__(**kwargs)

        self.kwargs |= {
            "n_shapelets": n_shapelets,
            "n_ts_for_selection": n_ts_for_selection,
            "n_features_strategy": n_features_strategy,
            "selection": selection,
            "distance": distance,
            "proximity_on_same_features": proximity_on_same_features,
            "scaler": scaler,
            "random_state": random_state,
            "use_combination": use_combination,
            "n_jobs": n_jobs,
        }

    def fit(self, X, y, idx=None, context=None, sample_weight=None, check_input=True):
        if self.scaler is not None:
            self.scaler.fit(X)

        if idx is None:
            idx = slice(None)

        self.y_lims = [X.min(), X.max()]

        X = X[idx]
        y = y[idx]
        if self.scaler is not None:
            X = self.scaler.transform(X)

        random.seed(self.random_state)
        if sample_weight is not None:
            pass
            #warnings.warn(f"sample_weight is not supported for {self.__class__.__name__}", Warning)

        self.st = TabularShapelets(n_shapelets=self.n_shapelets,
                                   n_ts_for_selection=self.n_ts_for_selection,
                                   n_features_strategy=self.n_features_strategy,
                                   selection=self.selection,
                                   distance=self.distance,
                                   use_combination=self.use_combination,
                                   random_state=random.randint(0, 2 ** 32 - 1),
                                   n_jobs=self.n_jobs
                                   )

        X_dist = self.st.fit_transform(X, y)
        X_bool, i_idx, j_idx = self._get_proximity_matrix(X_dist)

        X_bool, X_bool_idx = np.unique(X_bool, axis=1, return_index=True)
        #tqdm.write(f'Size reduced by {round((X_bool_shape[1] - X_bool.shape[1])/X_bool_shape[1]*100, 2)}%')

        super().fit(X_bool, y, context=context, sample_weight=sample_weight)
        if not hasattr(self, 'tree_') or self.tree_.feature[0] == -2:
            raise ValueError("No split found")
            #raise Exception("No split found")
        selected_pair = X_bool_idx[self.tree_.feature[0]]
        shapelet1_idx = i_idx[selected_pair]
        shapelet2_idx = j_idx[selected_pair]

        self.st._optimize_memory(np.array([shapelet1_idx, shapelet2_idx]))
        X_bool2, _, _ = self._get_proximity_matrix(self.st.transform(X, y))

        super().fit(X_bool2, y=y, sample_weight=sample_weight, check_input=check_input)

        return self

    def _get_proximity_matrix(self, X_dist):
        actual_n_shapelets = X_dist.shape[1]

        nan_masks = np.isnan(self.st.shapelets)  # shape: (n_shapelets, len_shapelet)
        same_index = (nan_masks[:, None] == nan_masks).all(axis=2)
        self.same_index = same_index

        if actual_n_shapelets < 2:
            raise ValueError(f"Hyperparameter led to n_shapelets={actual_n_shapelets}, it should be at least 2")

        tri_indices = np.triu_indices(actual_n_shapelets, k=1)
        if self.proximity_on_same_features:
            same_index_flat = self.same_index[tri_indices]
            i_idx = tri_indices[0][same_index_flat]
            j_idx = tri_indices[1][same_index_flat]

            X_bool = np.less_equal(X_dist[:, i_idx], X_dist[:, j_idx])
        else:
            i_idx, j_idx = tri_indices
            X_bool = np.less_equal(X_dist[:, i_idx], X_dist[:, j_idx])

        return X_bool, i_idx, j_idx

    def apply(self, X, check_input=False):
        """Apply the decision tree stump to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timepoints)
            The time series data to classify.
        check_input : bool, default=False
            Whether to check the input parameters.

        Returns
        -------
        X_leaves : array-like of shape (n_samples,)
            For each datapoint x in X, returns the leaf node it ends up in.
        """
        if self.scaler is not None:
            X = self.scaler.transform(X)

        self.y_lims = [min(self.y_lims[0], X.min()), min(self.y_lims[1], X.max())]
        X_dist = self.st.transform(X)
        actual_n_shapelets = X_dist.shape[1]

        i_idx, j_idx = np.triu_indices(actual_n_shapelets, k=1)
        X_bool = (X_dist[:, i_idx] <= X_dist[:, j_idx])

        return super().apply(X_bool, check_input=check_input)

    def supports(self, data_type):
        return data_type in [DATA_TYPE_TABULAR]

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

        f = self.feature_original[0]
        shapes_idx = None
        c = 0
        for i in range(self.st.shapelets.shape[0]):
            for j in range(i + 1, self.st.shapelets.shape[0]):
                if c == f:
                    shapes_idx = (i, j)
                c += 1

        with tempfile312.NamedTemporaryFile(delete_on_close=False,
                                            delete=False,
                                            suffix=".png",
                                            mode="wb") as temp_file:
            plt.figure(figsize=(2, 1))
            shape = self.st.shapelets[shapes_idx[0], 0]
            plt.plot([i for i in range(shape.shape[0])], shape, color="tab:green", alpha=0.7)
            shape = self.st.shapelets[shapes_idx[1], 0]
            plt.plot([i for i in range(shape.shape[0])], shape, color="tab:red", alpha=0.7)
            plt.ylim(*self.y_lims)
            plt.xlim(0, shape.shape[0])
            plt.gca().tick_params(axis='both', which='both', length=2, labelsize=6)
            plt.gca().spines['right'].set_color('none')
            plt.gca().spines['top'].set_color('none')
            # plt.gca().spines['bottom'].set_position('zero')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(temp_file, format="png", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        rule["textual_rule"] = f"{self.distance}(TS, green_shp)\n{comparison} {self.distance}(TS, red_shp)"
        rule["blob_rule"] = f"{self.distance}(TS, green_shp)\n{comparison} {self.distance}(TS, red_shp)"
        rule["graphviz_rule"] = {
            "image": f'{temp_file.name}',
            "imagescale": "true",
            "imagepos": "bc",
            "label": f"{self.distance}(TS, green_shp)\n\u2264 {self.distance}(TS, red_shp)",
            "labelloc": "t",
            "fixedsize": "true",
            "width": "2",
            "height": "1.33",
            "shape": "none",
            "fontsize": "8",
        }

        rule["not_textual_rule"] = f"{self.distance}(TS, green_shp)\n{not_comparison} {self.distance}(TS, red_shp)"
        rule["not_blob_rule"] = f"{self.distance}(TS, green_shp)\n{not_comparison} {self.distance}(TS, red_shp)"
        rule["not_graphviz_rule"] = {
            "image": f'{temp_file.name}',
            "imagescale": "true",
            "label": f"{self.distance}(TS, green_shp)\n{not_comparison} {self.distance}(TS, red_shp)",
            "imagepos": "bc",
            "labelloc": "t",
            "fixedsize": "true",
            "width": "2",
            "height": "1.33",
            "shape": "none",
            "fontsize": "8",
        }

        return rule

    def node_to_dict(self):
        """Convert the decision stump to a dictionary representation.

        This method is used for model serialization.

        Returns
        -------
        dict
            Dictionary representing the decision stump, including shapelet information.
        """
        rule = super().node_to_dict() | {
            'stump_type': self.__class__.__module__,
            "feature_idx": self.feature_original[0],
            "threshold": self.threshold_original[0],
            "is_categorical": self.is_categorical,
            "samples": self.n_node_samples[0]
        }

        rule["feature_name"] = f"Shapelet_{rule['feature_idx']}"

        comparison = "<="
        not_comparison = ">"

        rule["textual_rule"] = f"{self.distance}(TS, green_shp)\n{comparison} {self.distance}(TS, red_shp)"
        rule["blob_rule"] = f"{self.distance}(TS, green_shp)\n{comparison} {self.distance}(TS, red_shp)"
        rule["graphviz_rule"] = {
            "image": f'None',
            "imagescale": "true",
            "imagepos": "bc",
            "label": f"{self.distance}(TS, green_shp)\n{comparison} {self.distance}(TS, red_shp)",
            "labelloc": "t",
            "fixedsize": "true",
            "width": "2",
            "height": "1.33",
            "shape": "none",
            "fontsize": "8",
        }

        rule["not_textual_rule"] = f"{self.distance}(TS, green_shp)\n{not_comparison} {self.distance}(TS, red_shp)"
        rule["not_blob_rule"] = f"{self.distance}(TS, green_shp)\n{not_comparison} {self.distance}(TS, red_shp)"
        rule["not_graphviz_rule"] = {
            "image": f'{None}',
            "imagescale": "true",
            "label": f"{self.distance}(TS, green_shp)\n{not_comparison} {self.distance}(TS, red_shp)",
            "imagepos": "bc",
            "labelloc": "t",
            "fixedsize": "true",
            "width": "2",
            "height": "1.33",
            "shape": "none",
            "fontsize": "8",
        }

        # shapelet transform stuff
        rule["shapelets"] = self.st.shapelets.tolist()
        rule["n_shapelets"] = self.st.n_shapelets
        rule["n_shapelets_for_selection"] = self.st.n_shapelets_for_selection
        rule["n_ts_for_selection_per_class"] = self.st.n_ts_for_selection
        rule["sliding_window"] = self.st.sliding_window
        rule["selection"] = self.st.selection
        rule["distance"] = self.st.distance
        rule["mi_n_neighbors"] = self.st.mi_n_neighbors
        rule["random_state"] = self.st.random_state
        rule["n_jobs"] = self.st.n_jobs
        rule["y_lims"] = self.y_lims

        return rule

    @classmethod
    def dict_to_node(cls, node_dict, X=None):
        self = cls(
            n_shapelets=node_dict["n_shapelets"],
            n_shapelets_for_selection=node_dict["n_shapelets_for_selection"],
            n_ts_for_selection=node_dict["n_ts_for_selection_per_class"],
            sliding_window=node_dict["sliding_window"],
            selection=node_dict["selection"],
            distance=node_dict["distance"],
            mi_n_neighbors=node_dict["mi_n_neighbors"],
            random_state=node_dict["random_state"],
            n_jobs=node_dict["n_jobs"]
        )

        self.st = Shapelets(
            n_shapelets=node_dict["n_shapelets"],
            n_shapelets_for_selection=node_dict["n_shapelets_for_selection"],
            n_ts_for_selection_per_class=node_dict["n_ts_for_selection_per_class"],
            sliding_window=node_dict["sliding_window"],
            selection=node_dict["selection"],
            distance=node_dict["distance"],
            mi_n_neighbors=node_dict["mi_n_neighbors"],
            random_state=node_dict["random_state"],
            n_jobs=node_dict["n_jobs"]
        )

        self.st.shapelets = np.array(node_dict["shapelets"])

        self.feature_original = np.zeros(3, dtype=int)
        self.threshold_original = np.zeros(3)
        self.n_node_samples = np.zeros(3, dtype=int)

        self.y_lims = node_dict["y_lims"]

        self.feature_original[0] = node_dict["feature_idx"]
        self.threshold_original[0] = node_dict["threshold"]
        self.n_node_samples[0] = node_dict["samples"]
        self.is_categorical = node_dict["is_categorical"]

        args = copy.deepcopy(node_dict["args"])
        self.is_oblique = args.pop("is_oblique")
        self.is_pivotal = args.pop("is_pivotal")
        self.unique_val_enum = args.pop("unique_val_enum")
        self.coefficients = args.pop("coefficients")
        self.kwargs = args

        return self
