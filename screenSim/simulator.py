import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Simulator:
    def __init__(
        self, 
        num_genes: int = 20000,
        num_sgRNAs_per_gene = 5,
        num_control: int = 2, 
        num_treatment: int = 2,
        fraction_enriched: float = 0.2,
        fraction_depleted: float = 0.2,
        fraction_NTC: float = 0.2,
        min_total: int = 1e6,
        max_total: int = 1e8,
        lam_min: float = 0.44,
        lam_max: float = 1.44,
        p_min: float = 2.3e-3,
        p_max: float = 2.7e-3,
        lam_e_min: float = 1.2,
        lam_e_max: float = 2.0,
        lam_d_min: float = 0.2,
        lam_d_max: float = 1.0,
        p_e_min: float = 0.2,
        p_e_max: float = 1.0,
        p_d_min: float = 1.2,
        p_d_max: float = 2.0,
        type_dist: str = "poisson"):
        
        """
        Constructor for initializing Simulator object.
        
        Parameters
        ----------
        num_genes : int
            Number of genes.
        num_sgRNAs_per_gene : int
            Number of sgRNAs per gene.  
        num_treatment : int
            Number of treatment libraries.
        num_control : int
            Number of control libraries.
        fraction_enriched : float
            The fraction of enriched genes with respect to all genes. 
        fraction_depleted : float
            The fraction of depleted genes with respect to all genes. 
        fraction_NTC : float
            The fraction of NTC genes with respect to all genes.
        min_total : int
            The lower bound of the total number of counts for one library. 
        max_total : int
            The upper bound of the total number of counts for one library.
        type_dist : str
            Either "poisson" or "negative binomial" distribution. 
        
        """ 
        
        self.num_genes = int(num_genes)
        self.num_sgRNAs_per_gene = int(num_sgRNAs_per_gene)
        self.num_control = int(num_control)
        self.num_treatment = int(num_treatment)
        self.type_dist = type_dist

        self._init_fractions(fraction_enriched, fraction_depleted, fraction_NTC)
        self._init_totals_bounds(int(min_total), int(max_total))
        self._init_lam_bounds(lam_min, lam_max)
        self._init_p_bounds(p_min, p_max)
        self._init_e_lam(lam_e_min, lam_e_max)
        self._init_d_lam(lam_d_min, lam_d_max)
        self._init_e_p(p_e_min, p_e_max)
        self._init_d_p(p_d_min, p_d_max)
        
        self._num_sgRNAs()
        self._init_count_totals()
        self._split_genes()
        self._split_sgRNAs()
        self._init_sgRNA()
        self._init_gene()
        self._init_lambda()
        self._init_p()
        self._init_S_lam()
        self._init_S_p()
        self._mult_S_lam()
        self._mult_S_p()
        self._init_modification()
        
    def _init_fractions(self, e: float, d: float, ntc: float):
        """
        Initializes the enriched, depleted, NTC, and normal fractions.
        
        Raises
        ------
        Exception
            If the total of the fractions (enriched, depleted, NTC) exceeds 1. 
        
        """
        total = e + d + ntc
        
        if ((total > 0.0) & (total <= 1.0)):
            self.fraction_enriched = e
            self.fraction_depleted = d
            self.fraction_NTC = ntc
            self.fraction_normal = 1.0 - (e + d + ntc)
        else:
            raise Exception("Fractions total cannot exceed 1.")
       
    def _init_totals_bounds(self, lower: int, upper: int):
        """
        Initializes count totals bounds. 
        
        Raises
        ------
        Exception
            If lower/upper bounds <= 0 and/or lower > upper.
        
        """
        if ((lower < upper) & (lower > 0) & (upper > 0)):
            self.min_total = lower
            self.max_total = upper
        else:
            raise Exception("Lower/upper bounds must be positive and min_total should be less than max_total.")
            
    def _init_lam_bounds(self, lower: float, upper: float):
        """
        Initializes the enriched scalar bounds.
        
        Raises
        ------
        Exception
            If lower/upper bounds <= 0 and/or lower > upper.
        
        """
        if ((lower < upper) & (lower > 0) & (upper > 0)):
            self.lam_min = lower
            self.lam_max = upper
        else:
            raise Exception("Lower/upper bounds must be positive and scalar_e_min should be less than scalar_e_max.")
    
    def _init_p_bounds(self, lower: float, upper: float):
        
        if ((lower < upper) & (lower > 0) & (upper > 0)):
            self.p_min = lower
            self.p_max = upper
        else:
            raise Exception("Lower/upper bounds must be positive and scalar_e_min should be less than scalar_e_max.")
            
    def _init_e_lam(self, lower: float, upper: float):
        """
        Initializes the enriched scalar bounds.
        
        Raises
        ------
        Exception
            If lower/upper bounds <= 0 and/or lower > upper.
        
        """
        if ((lower < upper) & (lower > 0) & (upper > 0)):
            self.lam_e_min = lower
            self.lam_e_max = upper
        else:
            raise Exception("Lower/upper bounds must be positive and scalar_e_min should be less than scalar_e_max.")
     
    def _init_d_lam(self, lower: float, upper: float):
        """
        Initializes the depleted scalar bounds.
        
        Raises
        ------
        Exception
            If lower/upper bounds <= 0 and/or lower > upper.
        
        """
        if ((lower < upper) & (lower > 0) & (upper > 0)):
            self.lam_d_min = lower
            self.lam_d_max = upper
        else:
            raise Exception("Lower/upper bounds must be positive and scalar_d_min should be less than scalar_d_max.")
            
    def _init_e_p(self, lower: float, upper: float):
        """
        Initializes the enriched scalar bounds.
        
        Raises
        ------
        Exception
            If lower/upper bounds <= 0 and/or lower > upper.
        
        """
        if ((lower < upper) & (lower > 0) & (upper > 0)):
            self.p_e_min = lower
            self.p_e_max = upper
        else:
            raise Exception("Lower/upper bounds must be positive and scalar_e_min should be less than scalar_e_max.")
            
    def _init_d_p(self, lower: float, upper: float):
        """
        Initializes the enriched scalar bounds.
        
        Raises
        ------
        Exception
            If lower/upper bounds <= 0 and/or lower > upper.
        
        """
        if ((lower < upper) & (lower > 0) & (upper > 0)):
            self.p_d_min = lower
            self.p_d_max = upper
        else:
            raise Exception("Lower/upper bounds must be positive and scalar_e_min should be less than scalar_e_max.")
   
    def _num_sgRNAs(self):
        """
        Calculates total number of sgRNAs.  
        
        """
        self.num_sgRNAs = self.num_genes * self.num_sgRNAs_per_gene
        
    def _init_count_totals(self):
        """
        Initializes total sgRNA counts for each library. 
        
        """
        self.totals_array = np.random.randint(self.min_total, self.max_total, size = self.num_treatment + self.num_control) 
    
    def _split_genes(self):
        """
        Calculates number of enriched, depleted, and ntc genes based on fractions. 
        
        """
        self.num_g_e = round(self.num_genes * self.fraction_depleted)
        self.num_g_d = round(self.num_genes * self.fraction_depleted)
        self.num_g_ntc = round(self.num_genes * self.fraction_depleted)
        
    def _split_sgRNAs(self):
        """
        Calculates number of enriched, depleted, and ntc sgRNAs.
        
        """
        self.num_e = self.num_g_e * self.num_sgRNAs_per_gene
        self.num_d = self.num_g_d * self.num_sgRNAs_per_gene
        self.num_ntc = self.num_g_ntc * self.num_sgRNAs_per_gene    
    
    def _init_sgRNA(self):
        """
        Initializes array sgRNA labels. 
        
        """ 
        self.sgRNA = [f"sg_{i}" for i in np.arange(self.num_sgRNAs)]
    
    def _init_gene(self) -> list:
        """
        Initializes array with gene labels. 
        
        """
        gene = np.arange(self.num_genes)
        gene = np.repeat(gene, self.num_sgRNAs_per_gene)
        gene_label = [f"gene_{i}" for i in gene]
        
        ntc_genes = gene[self.num_e + self.num_d: self.num_e + self.num_d + self.num_ntc]
        gene_label[self.num_e + self.num_d: self.num_e + self.num_d + self.num_ntc] = [f"non-targeting_{i}" for i in ntc_genes]
        
        self.gene = gene_label
    
    def _init_lambda(self):
        """
        Initializes a lambda for each sgRNA.
        
        """
        self.lam = np.random.uniform(self.lam_min, self.lam_max, size = self.num_sgRNAs)

    def _init_p(self):
        """
        Initializes a probability for each sgRNA if the distribution is negative binomial.
        
        """
        if self.type_dist == "negative binomial":
            self.p = np.random.uniform(self.p_min, self.p_max, size = self.num_sgRNAs)
        else:
            self.p = 0
    
    def _init_S_lam(self):
        """
        Initializes gene-specific scalars for each gene. 
        
        """
        S = np.ones(self.num_sgRNAs)
        
        gene_e_scalars = np.random.uniform(self.lam_e_min, self.lam_e_max, size = self.num_g_e)
        gene_d_scalars = np.random.uniform(self.lam_d_min, self.lam_d_max, size = self.num_g_d)
        
        S[:self.num_e] = np.repeat(gene_e_scalars, self.num_sgRNAs_per_gene)
        S[self.num_e: self.num_e + self.num_d] = np.repeat(gene_d_scalars, self.num_sgRNAs_per_gene)
        
        self.S_lam = S 
        
    def _init_S_p(self):
        
        S = np.ones(self.num_sgRNAs)
        
        gene_e_scalars = np.random.uniform(self.p_e_min, self.p_e_max, size = self.num_g_e)
        gene_d_scalars = np.random.uniform(self.p_d_min, self.p_d_max, size = self.num_g_d)
        
        S[:self.num_e] = np.repeat(gene_e_scalars, self.num_sgRNAs_per_gene)
        S[self.num_e: self.num_e + self.num_d] = np.repeat(gene_d_scalars, self.num_sgRNAs_per_gene)
        
        self.S_p = S 
        
    def _mult_S_lam(self):
        """
        Scales the lambdas for treatment libraries by performing an element-wise product of `S` and `lam`.
            
        """
        self.S_x_lam = np.multiply(self.S_lam, self.lam)
    
    def _mult_S_p(self):
        
        self.S_x_p = np.multiply(self.S_p, self.p) 
     
    def _init_modification(self):
        """
        Labels each sgRNA as enriched, depleted, ntc, or normal. 
        
        """
        
        mod = ["normal"] * self.num_sgRNAs
        
        mod[:self.num_e] = ["enriched"] * self.num_e
        mod[self.num_e: self.num_e + self.num_d] = ["depleted"] * self.num_d
        mod[self.num_e + self.num_d: self.num_e + self.num_d + self.num_ntc] = ["ntc"] * self.num_ntc
        
        self.modification = mod
        
    def _sampling(self, lambdas: np.ndarray, p_array: np.ndarray) -> np.ndarray:
        """
        Generates count values for each lambda/p value given a distribution.
        
        Parameters
        ----------
        lambdas: np.ndarray
            To use as lam in poisson or n in negative binomial.
        p_array: np.ndarray 
            Probabilities to use as p in negative binomial.
            
        Raises
        ------
        Exception
            If input is not "poisson" or "negative binomial".
            
        Returns
        -------
        a : np.ndarray
            Count values for a given library. 
        
        """
        
        if self.type_dist == "poisson":
            a = np.random.poisson(lambdas)
        elif self.type_dist == "negative binomial":
            a = np.random.negative_binomial(lambdas, p_array)
        else:
            raise Exception("Make sure to choose a distribution from those available.")
            
        return a
        
    
    def _normalize(self, norm: np.ndarray, index: int) -> np.ndarray:
        """
        Adjusts array to have specified total.
        
        Parameters
        ----------
        norm: np.ndarray
            Array to normalize. 
        index : int
            Index to specify which total to use from `totals_array`.
        
        Returns
        -------
        norm : np.ndarray
            Array with a total of an elemnet from `totals_array`    
        
        """
        norm = norm.astype(float)
        norm /= (norm.sum())
        norm *= self.totals_array[index]
        norm = np.round(norm)
        
        return norm
    
    def _setting_control_libraries(self, control: pd.DataFrame):
        """
        Generates values for control libraries with _sum_array() and appends each library as a column to passed pd.DataFrame.
            
        """
        for i in np.arange(self.num_control):
            control[f"control_{i}"] = self._normalize(self._sampling(self.lam, self.p), i)
    
    def _setting_treatment_libraries(self, treatment: pd.DataFrame):
        """
        Generates values for treatment libraries with _sum_array() and appends each library as a column to passed pd.DataFrame.
            
        """
        for i in np.arange(self.num_treatment):
            treatment[f"treatment_{i}"] = self._normalize(self._sampling(self.S_x_lam, self.S_x_p), -(i+1))
    
    def sample(self, seed: int = 10) -> pd.DataFrame:
        """
        Generates DataFrame with observations for the simulation. 
        
        Parameters
        ----------
        seed: int
            Observations are repeatable each time sample() is called on 
            the same instance with the same `seed`. 
        
        Returns
        -------
        result : pd.DataFrame 
            sgRNA, gene, lambda, scalar, scaled lambda, modification, and each control and treatment library as columns
        """
        
        np.random.seed(seed)
        
        if self.type_dist == "poisson":
            result = pd.DataFrame({
            "sgRNA": self.sgRNA, 
            "gene": self.gene, 
            "lambda": self.lam,
            "lam scalar": self.S_lam, 
            "scaled lambda": self.S_x_lam,
            "modification": self.modification
        })
            
        elif self.type_dist == "negative binomial":
            result = pd.DataFrame({
                "sgRNA": self.sgRNA, 
                "gene": self.gene, 
                "lambda": self.lam,
                "lam scalar": self.S_lam, 
                "scaled lambda": self.S_x_lam,
                "p scalar": self.S_p,
                "scaled p": self.S_x_p,
                "modification": self.modification
            })

        self._setting_control_libraries(result)
        self._setting_treatment_libraries(result)
        
        return result 