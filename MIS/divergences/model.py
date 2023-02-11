from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np
import sys

from .heuristic import Heuristic

class MultipleImportanceSampling(ABC):

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
        self._batch_samples = 0
        self._total_samples = 0
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
    def _update_alphas(self, batch_samples: int) -> None:
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
            self._alphas[m] * ((self._sigma_prime[m] / self._batch_samples) 
            - ((self._mu_prime[m] / self._batch_samples) ** 2))
            for m in range(self._nsampling)
        )
    
    def _sampling_checker(self, sampling: Callable[[float], float], begin: float, end: float):
    
        sample = sampling()
        
        return sample if sample >= begin and sample < end \
            else self._sampling_checker(sampling, begin, end)

    def _sample(self) -> float:
        
        # increment the number of computed samples
        self._batch_samples += 1
        self._total_samples += 1
        
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
        
    
    def fit(self, samples: int, batch: int=1) -> float:
        
        # TODO: improve this part
        batch_ends = list(np.arange(0, samples, batch))
        
        for b in batch_ends:
            
            # restore the number of computed samples to 0
            if self._batch_variance:
                self._batch_samples = 0
            
            end_batch = min(b + batch, samples)
                
            for _ in range(b, end_batch):
                
                # sample and update alphas if needed
                self._update_alphas(batch)
                self._sample()
                
            # intermediate computation of variance
            self._prepare_variance() 
            
            print(f'[{min(b + batch, samples)} samples] variance: {self.variance:.4f}')
        
        
class BalanceImportance(MultipleImportanceSampling):    

    def __init__(self, to_integrate, samplings, pdfs, interval):
        super(BalanceImportance, self).__init__(to_integrate, samplings, pdfs, interval)
    
    def _update_alphas(self, batch_samples: int) -> None:
        pass
        
    def _compute_balance(self, alphas: list[float], pdfs: list[float]) -> float:
        
        return Heuristic.balance_divergence(alphas, pdfs)
    
    def _compute_weight(self, index: int, pdfs: list[float]) -> float:
        
        return Heuristic.divergence_balance_weight(index, self._alphas, pdfs)
    


class TsallisImportance(MultipleImportanceSampling):    

    def __init__(self, to_integrate, samplings, pdfs, interval, gamma=1):
        super(TsallisImportance, self).__init__(to_integrate, samplings, pdfs, interval)
        
        self._gamma = 1
        
        # TODO: specify for only 2 samplings method?
        
        # TODO: keep track of Tsallis data
        self._xi_sum = 0
        self._xi_prime_sum = 0
    
    def _update_alphas(self, batch_samples: int) -> None:
        
        # keep track of data if exists
        if self._total_samples > 0 and self._total_samples % batch_samples != 0:
            
            for m in range(self._nsampling):
                
                sample = self._f_track[m][-1]
                pdfs = self._pdf_track[m][-1] # last recorded pdfs
                alpha_probs = Heuristic.balance_divergence(self._alphas, pdfs)
                
                self._xi_sum += (sample / (alpha_probs ** (self._gamma + 1))) \
                    * (pdfs[0] - pdfs[1])
                    
                self._xi_prime_sum += (sample / (alpha_probs ** (self._gamma + 2))) \
                    * ((pdfs[0] - pdfs[1]) ** 2)
        
        # if first batch, then adapt number of samples for each sampling technique
        if self._total_samples < batch_samples:
            
            if batch_samples % self._nsampling != 0:
                raise AssertionError("Number of samples cannot be equally dispatched when initialized method")
            
            # TODO : make this part generic for other methods?
            # attach the correct alphas weights depending of number of samples
            samples_method = [ batch_samples / self._nsampling ] * np.arange(0, self._nsampling)
            
            method_index = list(samples_method).index(list(
                filter(lambda x: x <= self._total_samples, samples_method)
            )[-1])
            
            self._alphas = [ 0.99 if method_index == i else 0.01 / self._nsampling \
                for i in range(self._nsampling) ]
            
            
        elif self._total_samples % batch_samples == 0:
            # basic case of Tsallis method
            print('Next update of alphas')
            
            xi_alpha = self._xi_sum / batch_samples
            xi_prime_alpha = self._xi_prime_sum * \
                (-self._gamma / batch_samples)
                
            if xi_prime_alpha <= 0:
                xi_prime_alpha = sys.float_info.epsilon
                
            self._alphas[0] -= (xi_alpha / xi_prime_alpha)
            
            if self._alphas[0] <= 0:
                self._alphas[0] = sys.float_info.epsilon
            
            if self._alphas[0] >= 1:
                self._alphas[0] = 1 - sys.float_info.epsilon
                
            self._alphas[1] = 1 - self._alphas[0]
            
            print(self._alphas)
        
    def _compute_balance(self, alphas: list[float], pdfs: list[float]) -> float:
        
        return Heuristic.balance_divergence(alphas, pdfs)
    
    def _compute_weight(self, index: int, pdfs: list[float]) -> float:
        
        return Heuristic.divergence_balance_weight(index, self._alphas, pdfs)
    