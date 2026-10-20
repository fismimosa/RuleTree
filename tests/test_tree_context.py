"""What the stumps and their components know of the tree: the tree (context) and the node (node_id)."""
import json

import pytest

from RuleTree import RuleTreeClassifier, RuleTreeCluster, RuleTreeClusterClassifier, RuleTreeRegressor
from RuleTree.base.RuleTreeBaseStump import tree_arguments
from RuleTree.stumps.Stump import Stump
from RuleTree.stumps.classification import DecisionTreeStumpClassifier, PivotTreeStumpClassifier
from RuleTree.stumps.splitters import ClassificationSplitter
from RuleTree.stumps.transformers import MatrixTransformer
from RuleTree.tree.RuleTree import RuleTree
from RuleTree.utils.data_utils import json_numpy_encoder


def json_round_trip(model):
    return RuleTree.from_dict(json.loads(json.dumps(model.to_dict(), cls=json_numpy_encoder)))


# what the components receive at each fit: (component, tree, node id, the node is a leaf, its parent is split)
CALLS = []


def record(component, context, node_id):
    node = context.get_node_by_id(node_id) if context is not None else None
    CALLS.append((component, context, node_id, node is not None and node.is_leaf(),
                  node is not None and (node.parent is None or node.parent.stump is not None)))


class RecordingTransformer(MatrixTransformer):
    def fit(self, X, y, idx=None, data_type=None, cache=None, sample_weight=None, context=None, node_id=None):
        record("transformer", context, node_id)
        return super().fit(X, y, idx, data_type=data_type, cache=cache, sample_weight=sample_weight)


class RecordingSplitter(ClassificationSplitter):
    def fit(self, X, y, numerical, categorical, sample_weight=None, check_input=True, **kwargs):
        record("splitter", kwargs["context"], kwargs["node_id"])
        return super().fit(X, y, numerical, categorical, sample_weight, check_input)


@pytest.mark.parametrize("make_model, dataset", [
    (lambda: RuleTreeClassifier(max_depth=3, random_state=0), "iris"),
    (lambda: RuleTreeClassifier(max_depth=3, random_state=0, stump_selection="best",
                                base_stumps=[DecisionTreeStumpClassifier(), PivotTreeStumpClassifier()]), "iris"),
    (lambda: RuleTreeRegressor(max_depth=3, random_state=0), "diabetes"),
    (lambda: RuleTreeCluster(max_depth=3, random_state=0), "blobs"),
    (lambda: RuleTreeClusterClassifier(max_depth=3, random_state=0, component_split="joint"), "iris"),
], ids=["classifier", "classifier-best", "regressor", "cluster", "cluster-joint"])
def test_stumps_know_their_node(make_model, dataset, request):
    X, y = request.getfixturevalue(dataset)
    model = make_model().fit(X, y)

    for node_id, node in model.get_predicates().items():
        assert node.stump.node_id_ == node_id
    for node_id, node in json_round_trip(model).get_predicates().items():
        assert node.stump.node_id_ == node_id


def test_stump_outside_a_tree(iris):
    X, y = iris

    assert DecisionTreeStumpClassifier().node_id_ is None
    assert DecisionTreeStumpClassifier().fit(X, y).node_id_ is None


def test_trees_of_the_stumps_rename_the_node(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=3, random_state=0).fit(X, y)

    for tree in model.stumps_to_trees(model.get_balanced_stumps(p=0.1)).values():
        assert tree.root.stump.node_id_ == "R"
    assert all(node.stump.node_id_ == node_id for node_id, node in model.get_predicates().items())


def test_components_receive_the_tree_and_the_node(iris):
    X, y = iris
    CALLS.clear()

    model = RuleTreeClassifier(max_depth=2, random_state=0,
                               base_stumps=Stump(RecordingTransformer(), RecordingSplitter())).fit(X, y)

    # the pure nodes become leaves without fitting a stump
    assert {(component, node_id) for component, _, node_id, _, _ in CALLS} == \
        {(component, node_id) for component in ("transformer", "splitter") for node_id in model.get_predicates()}
    assert all(context is model and is_leaf and parent_is_split for _, context, _, is_leaf, parent_is_split in CALLS)

    CALLS.clear()
    Stump(RecordingTransformer(), RecordingSplitter()).fit(X, y)
    assert [(component, context, node_id) for component, context, node_id, _, _ in CALLS] == \
        [("transformer", None, None), ("splitter", None, None)]


def test_tree_arguments_only_if_declared():
    def without(X, y):
        pass

    def with_node(X, y, node_id=None):
        pass

    def with_kwargs(X, y, **kwargs):
        pass

    assert tree_arguments(without, "tree", "Rl") == {}
    assert tree_arguments(with_node, "tree", "Rl") == {"node_id": "Rl"}
    assert tree_arguments(with_kwargs, "tree", "Rl") == {"context": "tree", "node_id": "Rl"}
