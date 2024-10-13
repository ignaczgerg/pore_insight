import numpy as np

class MolarVolumeRelation:
    @staticmethod
    def relation_vs_a(x):
        return 1.1471 * x + 11.248

    @staticmethod
    def relation_vs_b(x):
        return -0.0002 * (x**2) + 1.3472 * x - 4.8372

    @staticmethod
    def new_relation(x):
        pass 