"""TfidfTransformer: texts as the TF-IDF weights of their words."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from RuleTree.base.RuleTreeBaseTransformer import RuleTreeBaseTransformer
from RuleTree.utils.define import DATA_TYPE_TEXT


class TfidfTransformer(RuleTreeBaseTransformer):
    """
    Transformer of texts into the TF-IDF weights of their words (scikit-learn's ``TfidfVectorizer``): one
    feature per word of the vocabulary, so that a split is a threshold on the weight of a word, e.g.
    ``goal <= 0.21``.

    The vocabulary and the inverse document frequencies are learned once, on all the texts given to the
    tree, and cached in it. Combined with a splitter, it makes a stump on texts, e.g.
    ``Stump(TfidfTransformer(), ClassificationSplitter())``.

    Parameters
    ----------
    max_features : int, default=1000
        Size of the vocabulary: the most frequent words. None for all the words.
    ngram_range : tuple (min_n, max_n), default=(1, 1)
        Lengths of the sequences of words of the vocabulary.
    stop_words : {'english'}, list or None, default=None
        Words ignored.
    min_df : int or float, default=1
        Minimum number (or fraction, if float) of texts in which a word occurs to be in the vocabulary.
    lowercase : bool, default=True
        Whether the texts are lowercased.

    Attributes
    ----------
    vocabulary_ : list of str
        The words, one per feature.
    idf_ : numpy.ndarray of shape (n_words,)
        Inverse document frequency of each word.
    numerical_ : list of int
        Numerical columns of the feature matrix (all of them).
    categorical_ : list of int
        Categorical columns of the feature matrix (none).

    Notes
    -----
    The weights of a text are normalized over all its words, so the whole vocabulary is kept (and saved)
    also after ``select``.

    Examples
    --------
    >>> import numpy as np
    >>> from RuleTree import RuleTreeClassifier
    >>> from RuleTree.stumps.Stump import Stump
    >>> from RuleTree.stumps.splitters import ClassificationSplitter
    >>> from RuleTree.stumps.transformers import TfidfTransformer
    >>> texts = np.array(["the team scored a goal", "a goal in the final", "the chef baked bread", "bread and oil"])
    >>> stump = Stump(TfidfTransformer(), ClassificationSplitter())
    >>> model = RuleTreeClassifier(max_depth=1, random_state=0, base_stumps=stump)
    >>> model = model.fit(X_txt=texts, y=["sport", "sport", "cooking", "cooking"])
    >>> model.root.stump.get_rule()["textual_rule"]
    'bread <= 0.227'
    """

    supported_data_types = (DATA_TYPE_TEXT,)

    def __init__(self, max_features=1000, ngram_range=(1, 1), stop_words=None, min_df=1, lowercase=True):
        """Initialize the TfidfTransformer; see the class documentation for the parameters."""
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.min_df = min_df
        self.lowercase = lowercase

    def _vectorizer(self, vocabulary=None):
        """A new TfidfVectorizer with the parameters of the transformer (and a fixed vocabulary, if given)."""
        return TfidfVectorizer(max_features=self.max_features, ngram_range=tuple(self.ngram_range),
                               stop_words=self.stop_words, min_df=self.min_df, lowercase=self.lowercase,
                               vocabulary=vocabulary)

    def fit(self, X, y, idx=None, data_type=None, sample_weight=None, *, context, node_id, cache):
        """
        Learn the vocabulary and the inverse document frequencies on all the texts (cached in the tree).

        Parameters
        ----------
        X : numpy.ndarray of str
            Texts, all the instances.
        y : numpy.ndarray
            Target, all the instances (not used).
        idx : numpy.ndarray, default=None
            Indices of the instances in the node (not used: the vocabulary is the one of all the texts).
        data_type : str, default=None
            Type of X, used as key of the cached values.
        sample_weight : numpy.ndarray, default=None
            Not used.
        context : RuleTree or None
            Tree being built, None outside a tree.
        node_id : str or None
            Identifier of the node, None outside a tree.
        cache : dict or None
            Cache of the tree (namespace -> dictionary of values), None outside a tree.

        Returns
        -------
        self : TfidfTransformer
        """
        cache = self._get_cache(cache)
        if ("tfidf", data_type) not in cache:
            vectorizer = self._vectorizer().fit([str(text) for text in X])
            cache[("tfidf", data_type)] = (vectorizer.get_feature_names_out().tolist(), vectorizer.idf_)

        self.vocabulary_, self.idf_ = cache[("tfidf", data_type)]
        self.vectorizer_ = self._vectorizer(self.vocabulary_)
        self.vectorizer_.idf_ = self.idf_
        self.numerical_, self.categorical_ = list(range(len(self.vocabulary_))), []
        return self

    def transform(self, X, context=None, node_id=None, cache=None):
        """
        TF-IDF weights of the words of the vocabulary in each text.

        Parameters
        ----------
        X : numpy.ndarray of str
            Texts.
        context : RuleTree, default=None
            The tree, None outside a tree.
        node_id : str, default=None
            Identifier of the node of the stump, None outside a tree.
        cache : dict, default=None
            Cache of the tree, when called by it; None otherwise.

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_words)
        """
        return self.vectorizer_.transform([str(text) for text in X]).toarray()

    def feature_id(self, feature):
        """
        The word of a feature.

        Parameters
        ----------
        feature : int
            Column of the feature matrix.

        Returns
        -------
        feature_id : str
        """
        return self.vocabulary_[feature]

    def get_state(self):
        """
        Fitted state: the vocabulary and the inverse document frequencies.

        Returns
        -------
        state : dict
            JSON-friendly dictionary, restored by set_state.
        """
        return {"vocabulary": list(self.vocabulary_), "idf": np.asarray(self.idf_).tolist()}

    def set_state(self, state):
        """
        Restore the fitted state saved by get_state.

        Parameters
        ----------
        state : dict
            Dictionary created by get_state.

        Returns
        -------
        self : TfidfTransformer
        """
        self.vocabulary_, self.idf_ = list(state["vocabulary"]), np.asarray(state["idf"])
        self.vectorizer_ = self._vectorizer(self.vocabulary_)
        self.vectorizer_.idf_ = self.idf_
        self.numerical_, self.categorical_ = list(range(len(self.vocabulary_))), []
        return self
