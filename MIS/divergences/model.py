from abc import ABC, abstractmethod
import numpy as np
from balance import Heuristic

class Model(ABC):

    def __init__(self, to_integrate, samplings, pdfs, interval):
        
        self._begin, self._end = interval
        self._f = to_integrate
        self._samplings = samplings
        self._pdfs = pdfs
        self._samples = 0
        
        if len(self._samplings) != self._pdfs:
            raise Exception('Require the same number of samplings methods \
                and their associate PDFs functions')
        
        self._alphas = np.full(
                shape=len(self._samplings),
                fill_value=1. / len(self._samplings),
                dtype=np.float32
            )
        
        self._f_track = [[] for i in range(self._samplings)]
        self._pdf_track = [[] for i in range(self._samplings)]
        
    @abstractmethod
    def _update_alphas(self):
        pass
    
    def _variance(self):
            
        v_sum_squared = 0
        v_sum = 0
        
        # TODO
        # for i, spp in enumerate(samples):
            
        #     v_sum_squared += (spp ** 2) / (bh_div(alpha, fpdfs[i], gpdfs[i]) ** 2)
        #     v_sum += (spp / bh_div(alpha, fpdfs[i], gpdfs[i]))
            
        # v_sum_squared /= len(samples)
        # v_sum /= len(samples)
        
        # return alpha * (v_sum_squared - (v_sum ** 2))
        
        return 0.
        
    def _sample(self):
        
        # increment the number of samples
        self._samples += 1
        
        # start computing the sample using MIS
        f_sum = 0
        
        for i, sampling_m in enumerate(self._samplings):
            
            x = sampling_m()
            
            pdfs = [ pdf_m(x) for pdf_m in self._pdfs ]
            
            f = self._f(x)
            
            # keep track of sums and pdfs
            self._f_track[i].append(f)
            self._pdf_track[i].append(pdfs) # TODO check how to store this part
            
            # TODO
            f_sum += f * Heuristic.balance_weight(i, pdfs)
            self._f_track
            
            f_sum += f
            
        # # sample using second method
        # x2 = sampling2()
        
        # # TODO: check reverse PDFs or not?
        # fpdf_2 = pdf_f2(x2)
        # gpdf_2 = pdf_f1(x2)
        
        # f2 = f(x2)
        
        # # use of MIS weight for sampling2 function
        # f_sum += f2 * bh_div_weight(1 - alpha, fpdf_2, gpdf_2)
        
        # final contribution is f_sum
        return f_sum
        
    
    def fit(self, _iterations, _batch=None):
        pass
        