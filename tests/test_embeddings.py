import numpy as np
import pytest

from RuleTree import RuleTreeClassifier
from RuleTree.stumps.classification import PivotTreeStumpClassifier, ProximityPivotTreeStumpClassifier
from RuleTree.utils.embeddings import ImageEmbedder, TextEmbedder

pytest.importorskip("torch")
pytest.importorskip("torchvision")

# random weights: nothing is downloaded
EMBEDDER = dict(weights=None, image_size=32, device="cpu")


@pytest.mark.parametrize("model, n_features", [("resnet18", 512), ("mobilenet_v3_small", 1024)])
def test_image_embedder_removes_the_classifier(model, n_features):
    embedder = ImageEmbedder(model, **EMBEDDER).fit()

    assert embedder.n_features_out_ == n_features
    assert embedder.transform(np.random.default_rng(0).random((3, 8, 8))).shape == (3, n_features)


@pytest.mark.parametrize("images", [np.random.default_rng(0).random((5, 12, 10)),
                                    np.random.default_rng(0).random((5, 12, 10, 1)),
                                    np.random.default_rng(0).random((5, 12, 10, 3)),
                                    np.random.default_rng(0).integers(0, 256, (5, 12, 10, 3), dtype=np.uint8)],
                         ids=["gray", "one channel", "rgb", "uint8"])
def test_image_embedder_shapes(images):
    X_embedded = ImageEmbedder(batch_size=2, **EMBEDDER).transform(images)  # fitted on first use

    assert X_embedded.shape == (5, 512)
    assert np.isfinite(X_embedded).all()


def test_image_embedder_batches_do_not_change_the_result():
    images = np.random.default_rng(0).random((7, 8, 8))
    embedder = ImageEmbedder(**EMBEDDER).fit()

    X_embedded = embedder.transform(images)

    assert np.allclose(embedder.set_params(batch_size=3).transform(images), X_embedded, atol=1e-5)


def test_image_embedder_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        ImageEmbedder(**EMBEDDER).transform(np.zeros((2, 8, 8, 2)))


def test_image_embedder_without_linear_layer():
    with pytest.raises(ValueError, match="linear"):
        ImageEmbedder("squeezenet1_0", weights=None, device="cpu").fit()  # a convolutional classifier


@pytest.mark.parametrize("stump", [PivotTreeStumpClassifier, ProximityPivotTreeStumpClassifier])
def test_pivot_tree_on_embedded_images(stump):
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 20)
    images = rng.random((40, 8, 8)) * 0.2
    images[y == 1, 2:6, 2:6] += 0.8  # a bright square in the class 1

    model = RuleTreeClassifier(max_depth=2, random_state=0, base_stumps=stump(embedder=ImageEmbedder(**EMBEDDER)))
    model.fit(X_img=images, y=y)

    assert np.mean(model.predict(X_img=images) == y) >= 0.9


@pytest.mark.slow  # downloads a small sentence transformer
def test_text_embedder():
    pytest.importorskip("sentence_transformers")
    texts = ["The striker scored a goal.", "The team won the final.", "The chef baked bread.",
             "Add garlic and olive oil."]

    embedder = TextEmbedder("all-MiniLM-L6-v2", device="cpu")
    X_embedded = embedder.fit_transform(texts)

    assert X_embedded.shape == (4, embedder.n_features_out_)
    assert np.allclose(np.linalg.norm(X_embedded, axis=1), 1)
    labels = np.array(["sport", "sport", "cooking", "cooking"])
    stump = PivotTreeStumpClassifier(embedder=TextEmbedder("all-MiniLM-L6-v2", device="cpu"))
    model = RuleTreeClassifier(max_depth=1, random_state=0, base_stumps=stump).fit(X_txt=np.array(texts), y=labels)
    assert np.array_equal(model.predict(X_txt=np.array(texts)), labels)
