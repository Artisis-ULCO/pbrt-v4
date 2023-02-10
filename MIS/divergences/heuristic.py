from __future__ import annotations
import numpy as np

class Heuristic():
    
    @staticmethod
    def balance(pdfs: list[float]):
        return sum(pdfs)
    
    @staticmethod
    def balance_divergence(alphas: list[float], pdfs: list[float]):
        return sum(np.asarray(alphas) * np.asarray(pdfs))
    
    @staticmethod
    def balance_weight(index: int, pdfs: list[float]):
        return pdfs[index] / sum(pdfs)
    
    @staticmethod
    def divergence_balance_weight(index: int, alphas: list[float], pdfs: list[float]):
        
        return (alphas[index] * pdfs[index]) \
                / sum(np.asarray(alphas) * np.asarray(pdfs))
        