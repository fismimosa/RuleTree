Citing RuleTree
===============

If RuleTree helps your research, please cite the library:

.. code-block:: bibtex

   @software{ruletree,
     title  = {{RuleTree}: interpretable decision trees built node by node with stumps of different kinds},
     author = {Landi, Cristiano and Cascione, Alessio and Guidotti, Riccardo},
     year   = {2025},
     url    = {https://github.com/fismimosa/RuleTree},
   }

If you use the pivot stumps, please also cite the papers that introduced them:

.. code-block:: bibtex

   @inproceedings{cascione2024pivot,
     title     = {Data-Agnostic Pivotal Instances Selection for Decision-Making Models},
     author    = {Cascione, Alessio and Setzu, Mattia and Guidotti, Riccardo},
     booktitle = {European Conference on Machine Learning and Principles and Practice of Knowledge
                  Discovery in Databases (ECML PKDD)},
     pages     = {367--386},
     year      = {2024},
     doi       = {10.1007/978-3-031-70341-6_22},
   }

   @misc{cascione2026proximity,
     title         = {Expanding Data-Agnostic Pivotal Instances Selection Models with Proximity Trees
                      and Ensemble Learning},
     author        = {Cascione, Alessio and Setzu, Mattia and Landi, Cristiano and
                      Mancarella, Paolo Maria and Guidotti, Riccardo},
     year          = {2026},
     eprint        = {2607.27522},
     archivePrefix = {arXiv},
     primaryClass  = {cs.LG},
   }

The second paper extends them to pairs of pivots (proximity splits) and to oblique splits.
