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
        min_total: int = 1e6,
        max_total: int = 1e8,
        total_NTCs: int = 1000,
        fraction_enriched: float = 0.2,
        fraction_depleted: float = 0.2,
        fraction_NTC: float = 0.2,
        scalar_e_min: float = 1.2,
        scalar_e_max: float = 2.0,
        scalar_d_min: float = 0.2,
        scalar_d_max: float = 1.0,
        type_dist: str = "poisson"):
        
        """
        Constructor for initializing Simulator object.
        
        Parameters
        ----------
        num_genes : int
            Number of genes.
        avg_num_sgRNAs : int
            Average number of sgRNAs across all genes. 
        num_treatment : int
            Number of treatment of libraries.
        num_control : int
            Number of control of libraries. 
        min_total : int
            The lower bound of the total number of counts for one library. 
        max_total : int
            The upper bound of the total number of counts for one library.
        total_NTCs : int
            Total number of non-targeting controls.
        fraction_enriched : float
            The fraction of enriched genes with respect to all genes. 
        fraction_depleted : float
            The fraction of depleted genes with respect to all genes. 
        fraction_NTC : float
            The fraction of NTC genes with respect to all genes.
        type_dist : str
            either "poisson" or "negative binomial" distribution. 
        
        """ 
        
        self.num_genes = int(num_genes)
        self.num_sgRNAs_per_gene = num_sgRNAs_per_gene
        self.num_control = int(num_control)
        self.num_treatment = int(num_treatment)
        self.min_total = int(min_total)
        self.max_total = int(max_total)
        self.total_NTCs = int(total_NTCs)
        self.scalar_e_min = scalar_e_min
        self.scalar_e_max = scalar_e_max
        self.scalar_d_min = scalar_d_min
        self.scalar_d_max = scalar_d_max
        self.type_dist = type_dist
        self.bounds = [10, 30]
        
        self._init_count_totals()
        self._init_fractions(fraction_enriched, fraction_depleted, fraction_NTC)
        self._num_sgRNA()
        self._split_genes()
        self._init_sgRNA()
        self._init_gene()
        self._init_lambda()
        self._init_p()
        self._init_S()
        self._init_S_l()
        self._init_modification()
     
    def _init_count_totals(self):
        """
        Initializes the totals of the sgRNA counts for each library. 
        
        """
        self.totals_array = np.random.randint(self.min_total, self.max_total, size = self.num_treatment + self.num_control)
     
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
    
    def _num_sgRNA(self):
        """
        Generates a number of sgRNAs per gene. 
        
        Returns
        -------
        sgRNAs : array
            The values of the array follow a normal distribution with the
            mean being `avg_num_sgRNAs`.
        
        """
        self.num_sgRNAs = self.num_genes * self.num_sgRNAs_per_gene
    
    def _split_genes(self):
        """
        Splits genes into enriched, depleted, ntc, or normal based on the fractions.
        
        """
        # sgRNAs
        self.num_e = round(self.num_sgRNAs * self.fraction_depleted)
        self.num_d = round(self.num_sgRNAs * self.fraction_depleted)
        self.num_ntc = round(self.num_sgRNAs * self.fraction_NTC)
        self.num_n = round(self.num_sgRNAs * self.fraction_normal)
        
        # genes
        self.num_g_e = round(self.num_genes * self.fraction_depleted)
        self.num_g_d = round(self.num_genes * self.fraction_depleted)
         
    
    def _init_sgRNA(self) -> list:
        """
        Generates list of numbered sgRNAs for use in sample() DataFrame.
        
        Returns
        ------
        list 
            All sgRNAs numbered.
        
        """ 
        self.sgRNA = np.arange(self.num_sgRNAs)
    
    def _init_gene(self) -> list:
        """
        Generates list of numbered genes for use in sample() DataFrame. 
        
        Returns
        -------
        list
            All genes numbered. Genes repeated for each of their sgRNAs. 
        
        """
        gene = np.arange(20000) 
        self.gene = np.repeat(gene, 5)
    
    def _init_lambda(self):
        """
        Initializes a lambda for each sgRNA.
        
        """
        self.lam = np.random.uniform(self.bounds[0], self.bounds[1], size = self.num_sgRNAs)

    def _init_p(self):
        """
        Initializes a p (probability for negative binomial) for each sgRNA.
        
        """
        self.p = np.random.random(size = self.num_sgRNAs)
    
    def _init_S(self):
        """
        Initializes gene-specific scalars for each gene. 
        
        """
        S = np.ones(self.num_sgRNAs)
        
        gene_e_scalars = np.random.uniform(self.scalar_e_min, self.scalar_e_max, size = self.num_g_e)
        
        gene_d_scalars = np.random.uniform(self.scalar_d_min, self.scalar_d_max, size = self.num_g_d)
        
        mask = np.arange(self.num_sgRNAs).reshape(self.num_genes, self.num_sgRNAs_per_gene)
        
        shift = gene_e_scalars.size
        
        for idx, scalar in enumerate(gene_e_scalars):
            S[mask[idx]] = scalar

        for idx, scalar in enumerate(gene_d_scalars):
            S[mask[idx + shift]] = scalar
    
        self.S = S 
        
    def _init_S_l(self) -> np.ndarray:
        """
        Scales the lambdas for treatment libraries. 
        
        Returns
        -------
        np.ndarray 
            Element-wise product of `S` and `lam`.  
            
        """
        self.S_l = np.multiply(self.S, self.lam)
     
    def _init_modification(self) -> list:
        """
        Labels genes as enriched, depleted, NTC, or normal.
        
        Returns
        -------
        type_of_change : list
            Strings of enriched, depleted, NTC, and normal for each gene, 
            based on the fractional representation specified upon 
            initialization.
            
        """
        
        modification = []
        
        e = ["enriched"] * self.num_e
        d = ["depleted"] * self.num_d
        ntc = ["ntc"] * self.num_ntc
        n = ["normal"] * self.num_n
        
        self.modification = e + d + ntc + n
    
    def _sum_array(self, index: int, lambdas: np.ndarray, p_array: np.ndarray) -> np.ndarray:
        """
        Creates an array of random integers with a specified sum.
        
        Parameters
        ----------
        index : int
            Index to specify which total to use from `totals_array`.
        lambdas: np.ndarray
            To use as lam in poisson or n in negative binomial.
        p_array: np.ndarray 
            Probabilities to use as p in negative binomial.
            
        Raises
        ------
        Exception
            If input is not "poisson" or "negative binomial"
            
        Returns
        -------
        a : array
            Randomly generated integers with sum of element from `totals_array`    
        
        """
        
        if self.type_dist == "poisson":
            a = np.random.poisson(lambdas)
        elif self.type_dist == "negative binomial":
            a = np.random.negative_binomial(lambdas, p_array)
        else:
            raise Exception("Make sure to choose a distribution from those available")
        
        a = a.astype(float)
        a /= (a.sum())
        a *= self.totals_array[index]
        a = np.round(a)
        
        return a
    
    def _setting_control_libraries(self, control: pd.DataFrame) -> list:
        """
        Generates values for control libraries.
        
        Returns
        -------
        control : list
            List of arrays, one for each library, generated by the _sum_array() method. 
            
        """
        for i in np.arange(self.num_control):
            control[f"control_{i}"] = self._sum_array(i, self.lam, self.p)
    
    def _setting_treatment_libraries(self, treatment: pd.DataFrame) -> list:
        """
        Generates values for treatment libraries.
        
        Returns
        -------
        treatment : list
            List of arrays, one for each library, generated by the _sum_array() method. 
            
        """
        for i in np.arange(self.num_treatment):
            treatment[f"treatment_{i}"] = self._sum_array(-(i+1), self.S_l, self.p)
    
            
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
        result : DataFrame 
            sgRNA, gene, lam, S_lam, control, treatments, and modification DataFrames concatenated   
            
        """
        
        np.random.seed(seed)
        
        result = pd.DataFrame({
            "sgRNA": self.sgRNA, 
            "gene": self.gene, 
            "lambda": self.lam,
            "scalar": self.S, 
            "scaled lambda": self.S_l,
            "modification": self.modification
        })
                    
        self._setting_control_libraries(result)
        self._setting_treatment_libraries(result)
        
        return result 