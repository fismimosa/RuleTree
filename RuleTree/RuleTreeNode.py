class RuleTreeNode():

    def __init__(self, idx, node_id, label, parent_id, is_leaf=False, clf=None, node_l=None, node_r=None,
                 samples=None, support=None, impurity=None, is_oblique=None, proba=None):
        self.idx = idx
        self.node_id = node_id
        self.label = label
        self.is_leaf = is_leaf
        self.clf = clf
        self.node_l = node_l
        self.node_r = node_r
        self.samples = samples
        self.support = support
        self.impurity = impurity
        self.is_oblique = is_oblique
        self.parent_id = parent_id
        self.medoid = None
        self.task_medoid = None
        self.proba = proba