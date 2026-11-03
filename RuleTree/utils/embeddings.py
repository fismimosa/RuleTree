"""
Embeddings of images and texts with pretrained neural networks, to train the pivot stumps (and any other
stump of tabular data) on them.

The pivot stumps compare instances by their distance, so images and texts are first mapped to vectors,
as in the PivotTree paper: the representation of an image is the output of the penultimate layer of a
convolutional network (e.g. ResNet or VGG), the one of a text the embedding of a sentence transformer.
Both transformers follow the scikit-learn interface, so they can precede a tree in a ``Pipeline``. They
need the ``embeddings`` extra (PyTorch, torchvision and sentence-transformers); the weights of the
networks are downloaded the first time.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from RuleTree.utils.optional import import_optional

# mean and standard deviation of the channels of ImageNet, used by the pretrained networks of torchvision
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _device(device):
    """
    The torch device to use: the given one, or the first available among CUDA, Apple MPS and the CPU.

    Parameters
    ----------
    device : str or None
        Device, e.g. ``"cpu"`` or ``"cuda"``; None to choose it.

    Returns
    -------
    device : str
    """
    if device is not None:
        return device

    torch = import_optional("torch", "The embeddings")
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ImageEmbedder(TransformerMixin, BaseEstimator):
    """
    Representation of images as the output of the penultimate layer of a convolutional network of
    torchvision, e.g. a ResNet or a VGG pretrained on ImageNet.

    The images are converted to three channels, resized to ``image_size`` and normalized as ImageNet;
    the last linear layer of the network (the classifier) is removed, so the output is the vector the
    network classifies, e.g. 512 values for ``resnet18`` and 4096 for ``vgg16``.

    Parameters
    ----------
    model : str, default='resnet18'
        Name of a classification network of torchvision (``torchvision.models.list_models``), e.g.
        ``'resnet50'``, ``'vgg16'``, ``'efficientnet_b0'``.
    weights : str or None, default='DEFAULT'
        Weights of the network, as accepted by ``torchvision.models.get_model``: ``'DEFAULT'`` for the
        best weights available, None for a network with random weights.
    image_size : int, default=224
        Side of the square images given to the network.
    batch_size : int, default=64
        Number of images processed at once.
    device : str, default=None
        Torch device, e.g. ``'cpu'`` or ``'cuda'``. If None, CUDA, Apple MPS or the CPU, the first
        available.

    Attributes
    ----------
    network_ : torch.nn.Module
        The network without its last linear layer, in evaluation mode.
    n_features_out_ : int
        Length of the representation of an image.

    See Also
    --------
    TextEmbedder : the same for texts.

    Notes
    -----
    The images are arrays of shape (n_images, height, width) for grayscale images, or
    (n_images, height, width, channels) with 1 or 3 channels (channels last). Integer images are in
    [0, 255], float images in [0, 1].

    The PivotTree paper uses networks trained on each data set: a network pretrained on ImageNet is a
    general-purpose representation, which can be fine-tuned outside RuleTree and passed as ``model``
    by name only if registered in torchvision.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.pipeline import make_pipeline
    >>> from RuleTree import RuleTreeClassifier
    >>> from RuleTree.stumps.classification import PivotTreeStumpClassifier
    >>> from RuleTree.utils.embeddings import ImageEmbedder
    >>> digits = load_digits()
    >>> model = make_pipeline(ImageEmbedder("resnet18"),  # doctest: +SKIP
    ...                       RuleTreeClassifier(max_depth=2, base_stumps=PivotTreeStumpClassifier()))
    >>> model = model.fit(digits.images / 16, digits.target)  # doctest: +SKIP
    """

    def __init__(self, model="resnet18", weights="DEFAULT", image_size=224, batch_size=64, device=None):
        """Initialize the ImageEmbedder; see the class documentation for the parameters."""
        self.model = model
        self.weights = weights
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device

    def fit(self, X=None, y=None):
        """
        Load the network (the weights are downloaded the first time). The images are not used.

        Parameters
        ----------
        X : numpy.ndarray, default=None
            Not used.
        y : numpy.ndarray, default=None
            Not used.

        Returns
        -------
        self : ImageEmbedder
        """
        torch = import_optional("torch", "ImageEmbedder")
        models = import_optional("torchvision.models", "ImageEmbedder")

        network = models.get_model(self.model, weights=self.weights)
        _remove_last_linear(network, torch.nn)
        self.network_ = network.eval().to(_device(self.device))
        self.n_features_out_ = int(self._embed(np.zeros((1, 8, 8), dtype=np.float32)).shape[1])
        return self

    def transform(self, X):
        """
        Representation of the images.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_images, height, width) or (n_images, height, width, channels)
            Images (see the notes of the class).

        Returns
        -------
        X_embedded : numpy.ndarray of shape (n_images, n_features_out)
        """
        if not hasattr(self, "network_"):
            self.fit()

        X = np.asarray(X)
        return np.vstack([self._embed(X[start:start + self.batch_size]) for start in range(0, len(X), self.batch_size)])

    def _embed(self, images):
        """
        Representation of a batch of images.

        Parameters
        ----------
        images : numpy.ndarray
            Batch of images (see the notes of the class).

        Returns
        -------
        X_embedded : numpy.ndarray of shape (n_images, n_features_out)
        """
        torch = import_optional("torch", "ImageEmbedder")

        images = np.asarray(images)
        if np.issubdtype(images.dtype, np.integer):
            images = images / 255.0
        if images.ndim == 3:
            images = images[..., None]
        if images.ndim != 4 or images.shape[-1] not in (1, 3):
            raise ValueError(f"Images of shape {images.shape[1:]}: expected (height, width) or (height, width, 1 or 3)")

        device = next(self.network_.parameters()).device
        batch = torch.as_tensor(images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        batch = batch.expand(-1, 3, -1, -1) if batch.shape[1] == 1 else batch
        batch = torch.nn.functional.interpolate(batch, size=(self.image_size, self.image_size), mode="bilinear",
                                                align_corners=False)
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

        with torch.no_grad():
            return self.network_((batch - mean) / std).flatten(1).cpu().numpy().astype(float)


def _remove_last_linear(network, nn):
    """
    Replace the last linear layer of a network (its classifier) with the identity.

    Parameters
    ----------
    network : torch.nn.Module
        The network, changed in place.
    nn : module
        ``torch.nn``.

    Raises
    ------
    ValueError
        If the network has no linear layer.
    """
    linear = [name for name, module in network.named_modules() if isinstance(module, nn.Linear)]
    if not linear:
        raise ValueError(f"{type(network).__name__} has no linear layer to remove")

    *parents, name = linear[-1].split(".")
    parent = network
    for part in parents:
        parent = getattr(parent, part)
    setattr(parent, name, nn.Identity())


class TextEmbedder(TransformerMixin, BaseEstimator):
    """
    Representation of texts as the embeddings of a sentence transformer (sentence-transformers), the
    representation used for the texts in the PivotTree paper (``all-mpnet-base-v2``).

    Parameters
    ----------
    model : str, default='all-mpnet-base-v2'
        Name or path of a model of sentence-transformers, e.g. ``'all-MiniLM-L6-v2'`` (smaller and
        faster).
    normalize : bool, default=True
        Whether the embeddings have unit length: then the Euclidean distance orders the texts as the
        cosine distance.
    batch_size : int, default=32
        Number of texts processed at once.
    device : str, default=None
        Torch device, e.g. ``'cpu'`` or ``'cuda'``. If None, CUDA, Apple MPS or the CPU, the first
        available.

    Attributes
    ----------
    encoder_ : sentence_transformers.SentenceTransformer
        The model.
    n_features_out_ : int
        Length of the embedding of a text.

    See Also
    --------
    ImageEmbedder : the same for images.

    Examples
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from RuleTree import RuleTreeClassifier
    >>> from RuleTree.stumps.classification import PivotTreeStumpClassifier
    >>> from RuleTree.utils.embeddings import TextEmbedder
    >>> texts = ["The striker scored a goal.", "The chef baked bread.", "The team won.", "Add garlic and oil."]
    >>> model = make_pipeline(TextEmbedder("all-MiniLM-L6-v2"),  # doctest: +SKIP
    ...                       RuleTreeClassifier(max_depth=1, base_stumps=PivotTreeStumpClassifier()))
    >>> model = model.fit(texts, ["sport", "cooking", "sport", "cooking"])  # doctest: +SKIP
    """

    def __init__(self, model="all-mpnet-base-v2", normalize=True, batch_size=32, device=None):
        """Initialize the TextEmbedder; see the class documentation for the parameters."""
        self.model = model
        self.normalize = normalize
        self.batch_size = batch_size
        self.device = device

    def fit(self, X=None, y=None):
        """
        Load the model (it is downloaded the first time). The texts are not used.

        Parameters
        ----------
        X : array-like of str, default=None
            Not used.
        y : numpy.ndarray, default=None
            Not used.

        Returns
        -------
        self : TextEmbedder
        """
        sentence_transformers = import_optional("sentence_transformers", "TextEmbedder")

        self.encoder_ = sentence_transformers.SentenceTransformer(self.model, device=_device(self.device))
        self.n_features_out_ = int(self.encoder_.get_sentence_embedding_dimension())
        return self

    def transform(self, X):
        """
        Embeddings of the texts.

        Parameters
        ----------
        X : array-like of str of shape (n_texts,)
            Texts.

        Returns
        -------
        X_embedded : numpy.ndarray of shape (n_texts, n_features_out)
        """
        if not hasattr(self, "encoder_"):
            self.fit()

        return self.encoder_.encode([str(text) for text in X], batch_size=self.batch_size,
                                    normalize_embeddings=self.normalize, convert_to_numpy=True).astype(float)
