from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np

from .heuristic import Heuristic

class MIS(ABC):

    def __init__(self, to_integrate: Callable[[float], float], 
                samplings: list[Callable[[float], float]], 
                pdfs: list[Callable[[float], float]], 
                interval: tuple[float, float],
                batch_variance=True):
        
        self._begin, self._end = interval
        self._f = to_integrate
        self._samplings = samplings
        self._pdfs = pdfs
        self._batch_variance = batch_variance
        self._track_samples = 0
        self._nsampling = len(self._samplings)
        
        if self._nsampling != len(self._pdfs):
            raise Exception('Require the same number of samplings methods \
                and their associate PDFs functions')
        
        self._weights = [ 1. / self._nsampling for _ in range(self._nsampling)]
        self._alphas = self._weights
        self._f_track = [ [] for _ in range(self._nsampling) ]
        self._pdf_track = [ [] for _ in range(self._nsampling) ]
        
        # track variance of F
        self._sigma_prime = None
        self._mu_prime = None
        
    @property
    def alphas(self) -> list[float]:
        return self._alphas
    
    @abstractmethod
    def _update_alphas(self) -> None:
        pass
    
    @abstractmethod
    def _compute_balance(self, alphas: list[float], pdfs: list[float]) -> float:
        pass
    
    @abstractmethod
    def _compute_weight(self, index: int, pdfs: list[float]) -> float:
        pass
    
    def _prepare_variance(self) -> None:
        
        # clear data when calling variance score computation
        if self._batch_variance:
            self._sigma_prime = [ 0 for _ in range(self._nsampling) ]
            self._mu_prime = [ 0 for _ in range(self._nsampling) ]
        
        for m in range(self._nsampling):
            
            for i, sample in enumerate(self._f_track[m]):
                
                bl_denominator = self._compute_balance(self._alphas, self._pdf_track[m][i])
                self._sigma_prime[m] += (sample ** 2) / (bl_denominator ** 2)
                self._mu_prime[m] += (sample / bl_denominator)
                
        # clear samples and pdf_tracks for next iteration
        self._f_track = [ [] for _ in range(self._nsampling) ]
        self._pdf_track = [ [] for _ in range(self._nsampling) ]
        
    @property
    def variance(self) -> float:
        """Compute the current obtained variance for the model
        Can be from scratch (all samples) or by batch
        
        Returns:
            float: computed variance
        """
        
        return sum(
            self._alphas[m] * ((self._sigma_prime[m] / self._track_samples) 
            - ((self._mu_prime[m] / self._track_samples) ** 2))
            for m in range(self._nsampling)
        )
    
    def _sampling_checker(self, sampling: Callable[[float], float], begin: float, end: float):
    
        sample = sampling()
        
        return sample if sample >= begin and sample < end \
            else self._sampling_checker(sampling, begin, end)

    def _sample(self) -> float:
        
        # increment the number of computed samples
        self._track_samples += 1
        
        # start computing the sample using MIS
        f_sum = 0
        
        for i, sampling_m in enumerate(self._samplings):
            
            sample = self._sampling_checker(sampling_m, self._begin, self._end)
            
            pdfs = [ pdf_m(sample) for pdf_m in self._pdfs ]
            
            f_sample = self._f(sample)
            
            # keep track of sums and pdfs
            self._f_track[i].append(f_sample)
            self._pdf_track[i].append(pdfs) 
            
            # compute the weight of this sampling method
            weight = self._compute_weight(i, pdfs)
            f_sum += f_sample * weight
            
            f_sum += f_sample
            
        # final contribution is f_sum
        return f_sum
        
    
    def fit(self, iterations: int, batch: int=1) -> float:
        
        # TODO: improve this part
        batch_ends = list(np.arange(0, iterations, batch))
        
        for b in batch_ends:
            
            # restore the number of computed samples to 0
            if self._batch_variance:
                self._track_samples = 0
            
            for _ in range(b, min(b + batch, iterations)):
                self._sample()
                
            # intermediate computation of variance
            self._prepare_variance() 
            
            print(f'[{min(b + batch, iterations)} samples] variance: {self.variance:.4f}')
            
            self._update_alphas()
        
        
class MIS_Tsallis(MIS):    

    def __init__(self, to_integrate, samplings, pdfs, interval):
        super(MIS_Tsallis, self).__init__(to_integrate, samplings, pdfs, interval)
    
    def _update_alphas(self) -> None:
        pass
        
    def _compute_balance(self, alphas: list[float], pdfs: list[float]) -> float:
        
        return Heuristic.balance_divergence(alphas, pdfs)
    
    def _compute_weight(self, index: int, pdfs: list[float]) -> float:
        
        return Heuristic.divergence_balance_weight(index, self._alphas, pdfs)
    