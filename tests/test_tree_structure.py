import pytest

from RuleTree import RuleTreeClassifier
from RuleTree.tree.RuleTree import RuleTree


@pytest.fixture
def model(iris):
    X, y = iris
    return RuleTreeClassifier(max_depth=3).fit(X, y)


def test_get_node_by_id(model):
    nodes = model.get_predicates() | model.get_leaf_nodes()

    assert len(nodes) > 3
    for node_id, node in nodes.items():
        assert model.get_node_by_id(node_id) is node


def test_get_node_by_id_missing(model):
    assert model.get_node_by_id("Rlllllll") is None


def test_leaves_and_predicates(model):
    leaves = model.get_leaf_nodes()
    predicates = model.get_predicates()

    assert set(leaves).isdisjoint(predicates)
    assert len(leaves) == len(predicates) + 1
    assert all(node.is_leaf() for node in leaves.values())
    assert not any(node.is_leaf() for node in predicates.values())


def test_get_rules(model):
    rules = model.get_rules()

    assert rules["node_id"] == "R"
    assert not rules["is_leaf"]
    assert rules["textual_rule"].startswith("X_")
    assert rules["left_node"]["node_id"] == "Rl"
    assert rules["right_node"]["node_id"] == "Rr"


def test_get_rules_column_names(model):
    names = ["sepal length", "sepal width", "petal length", "petal width"]

    rules = model.get_rules(columns_names=names)

    assert rules["feature_name"] in names


def test_print_rules(model, capsys):
    RuleTree.print_rules(model.get_rules())

    output = capsys.readouterr().out
    assert "|---" in output
    assert "output:" in output


def test_export_graphviz(model):
    pytest.importorskip("graphviz")

    graph = model.root.export_graphviz()

    assert "R ->" in graph.source
