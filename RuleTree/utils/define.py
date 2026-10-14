MODEL_TYPE_CLF = 'clf'
MODEL_TYPE_REG = 'reg'
MODEL_TYPE_CLU = 'clu'

DATA_TYPE_TABULAR = 'tabular'
DATA_TYPE_TS = 'ts'
DATA_TYPE_IMAGE = 'image'
DATA_TYPE_TEXT = 'text'

# keyword argument of fit/predict -> type of the data it contains
DATA_INPUT_TYPES = {
    'X': DATA_TYPE_TABULAR,
    'X_ts': DATA_TYPE_TS,
    'X_img': DATA_TYPE_IMAGE,
    'X_txt': DATA_TYPE_TEXT,
}

_GRAPHVIZ_DEFAULT_NODE = {
}

GRAPHVIZ_DEFAULT_NODE_SPLIT = _GRAPHVIZ_DEFAULT_NODE | {
    "shape": "box"
}

GRAPHVIZ_DEFAULT_NODE_LEAF = _GRAPHVIZ_DEFAULT_NODE | {
}
