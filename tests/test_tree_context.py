"""What the stumps and their components know of the tree: the tree (context) and the node (node_id)."""
import json

import pytest

from RuleTree import RuleTreeClassifier, RuleTreeCluster, RuleTreeClusterClassifier, RuleTreeRegressor
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
    def fit(self, X, y, idx=None, data_type=None, sample_weight=None, *, context, node_id):
        record("transformer", context, node_id)
        return super().fit(X, y, idx, data_type=data_type, sample_weight=sample_weight, context=context,
                           node_id=node_id)

    def transform(self, X, context=None, node_id=None):
        record("transform", context, node_id)
        return super().transform(X, context=context, node_id=node_id)


class RecordingSplitter(ClassificationSplitter):
    def fit(self, X, y, numerical, categorical, sample_weight=None, check_input=True, *, context, node_id):
        record("splitter", context, node_id)
        return super().fit(X, y, numerical, categorical, sample_weight, check_input, context=context, node_id=node_id)

    def update_statistics(self, X, y, sample_weight=None, *, context, node_id):
        record("update_statistics", context, node_id)
        return super().update_statistics(X, y, sample_weight, context=context, node_id=node_id)

    def apply(self, X, context=None, node_id=None):
        record("apply", context, node_id)
        return super().apply(X, context=context, node_id=node_id)


def fitted_calls(component_names):
    return [call for call in CALLS if call[0] in component_names]


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
    assert DecisionTreeStumpClassifier().fit(X, y, context=None, node_id=None).node_id_ is None


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

    fits = fitted_calls(("transformer", "splitter"))
    # the pure nodes become leaves without fitting a stump
    assert {(component, node_id) for component, _, node_id, _, _ in fits} == \
        {(component, node_id) for component in ("transformer", "splitter") for node_id in model.get_predicates()}
    assert all(context is model and is_leaf and parent_is_split for _, context, _, is_leaf, parent_is_split in fits)

    CALLS.clear()
    Stump(RecordingTransformer(), RecordingSplitter()).fit(X, y, context=None, node_id=None)
    assert [call[:3] for call in fitted_calls(("transformer", "splitter"))] == \
        [("transformer", None, None), ("splitter", None, None)]


def test_update_statistics_receives_the_tree_and_the_node(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2, random_state=0,
                               base_stumps=Stump(RecordingTransformer(), RecordingSplitter())).fit(X, y)
    CALLS.clear()

    model.update_statistics(X, y)

    updates = fitted_calls(("update_statistics",))
    assert len(updates) == len(model.get_predicates())
    assert {node_id for _, _, node_id, _, _ in updates} == set(model.get_predicates())
    assert all(context is model for _, context, _, _, _ in updates)


@pytest.mark.parametrize("predict", ["predict", "predict_proba", "apply"])
def test_predictions_pass_the_tree_and_the_node(predict, iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2, random_state=0,
                               base_stumps=Stump(RecordingTransformer(), RecordingSplitter())).fit(X, y)
    CALLS.clear()

    getattr(model, predict)(X)

    for component in ("transform", "apply"):
        calls = fitted_calls((component,))
        assert {node_id for _, _, node_id, _, _ in calls} == set(model.get_predicates())
        assert all(context is model for _, context, _, _, _ in calls)


def test_context_and_node_id_are_optional_in_the_predictions(iris):
    X, y = iris
    model = RuleTreeClassifier(max_depth=2, random_state=0,
                               base_stumps=Stump(RecordingTransformer(), RecordingSplitter())).fit(X, y)
    CALLS.clear()

    labels = model.root.stump.apply(X)

    assert set(labels) == {1, 2}
    assert [call[:3] for call in CALLS] == [("transform", None, None), ("apply", None, None)]


def test_context_and_node_id_are_required(iris):
    X, y = iris
    stump = DecisionTreeStumpClassifier().fit(X, y, context=None, node_id=None)

    with pytest.raises(TypeError, match="context"):
        DecisionTreeStumpClassifier().fit(X, y)
    with pytest.raises(TypeError, match="context"):
        stump.update_statistics(X, y)
    with pytest.raises(TypeError, match="context"):
        MatrixTransformer().fit(X, y)
    with pytest.raises(TypeError, match="context"):
        ClassificationSplitter().fit(X, y, [0, 1, 2, 3], [])
    with pytest.raises(TypeError, match="context"):
        stump.splitter_.update_statistics(X, y)
