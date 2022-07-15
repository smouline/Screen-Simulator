import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Simulator:
    def __init__(
        self, 
        num_genes: int = 20000, 
        avg_num_sgRNAs: int = 5,  
        num_control: int = 2, 
        num_treatment: int = 2,
        min_total: int = 1e6,
        max_total: int = 1e8,
        total_NTCs: int = 1000,
        fraction_enriched: float = 0.2,
        fraction_depleted: float = 0.2,
        fraction_NTC: float = 0.2
        scalar_e_min: float = 100.0,
        scalar_e_max: float = 1000.0,
        scalar_d_min: float = 0.001,
        scalar_d_max: float = 0.01,
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
        self.avg_num_sgRNAs = int(avg_num_sgRNAs)
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
        self._init_num_sgRNAs()
        self._split_genes()
        self._init_lambda()
        self._init_p()
        self._init_S()
     
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
    
    def _init_num_sgRNAs(self):
        """
        Generates a number of sgRNAs per gene. 
        
        Returns
        -------
        sgRNAs : array
            The values of the array follow a normal distribution with the
            mean being `avg_num_sgRNAs`.
        
        """
        sgRNAs = np.random.normal(loc=self.avg_num_sgRNAs, scale=1, size=self.num_genes)
        sgRNAs = np.round(sgRNAs)
        self.sgRNAs = sgRNAs 
    
    def _split_genes(self):
        """
        Splits genes into enriched, depleted, ntc, or normal based on the fractions.
        
        """
        num_e = round(len(self.sgRNAs) * self.fraction_enriched)
        num_d = round(len(self.sgRNAs) * self.fraction_depleted)
        num_ntc = round(len(self.sgRNAs) * self.fraction_NTC)
        num_n = round(len(self.sgRNAs) * self.fraction_normal)
        
        self.g_e = self.sgRNAs[0: num_e]
        self.g_d = self.sgRNAs[num_e: num_e + num_d]
        self.g_ntc = self.sgRNAs[num_e + num_d: num_e + num_d + num_ntc]
        self.g_n = self.sgRNAs[num_e + num_d + num_ntc: num_e + num_d + num_ntc + num_n]
    
    def _init_lambda(self):
        """
        Initializes a lambda for each sgRNA.
        
        """
        self.lam = np.random.uniform(self.bounds[0], self.bounds[1], size = int(self.sgRNAs.sum()))

    def _init_p(self):
        """
        Initializes a p (probability for negative binomial) for each sgRNA.
        
        """
        self.p = np.random.random(size = int(self.sgRNAs.sum()))
    
    def _init_S(self):
        """
        Initializes gene-specific scalars for each gene. 
        
        """
        S = []

        for i in self.g_e:
            g_scalar = np.random.uniform(self.scalar_e_min, self.scalar_e_max)
            for n in np.arange(i):
                S.append(g_scalar)

        for i in self.g_d:
            g_scalar = np.random.uniform(self.scalar_d_min, self.scalar_d_max)
            for n in np.arange(i):
                S.append(g_scalar)
                
        for i in self.g_ntc:
            for n in np.arange(i):
                S.append(1)
                
        for i in self.g_n:
            for n in np.arange(i):
                S.append(1)
            
        self.S = S 
    
    def _sgRNAs(self) -> list:
        """
        Generates list of numbered sgRNAs for use in sample() DataFrame.
        
        Returns
        ------
        list 
            All sgRNAs numbered.
        
        """
        return ["sgRNA_" + str(int(i)) for i in np.arange(self.sgRNAs.sum())]
    
    def _gene(self) -> list:
        """
        Generates list of numbered genes for use in sample() DataFrame. 
        
        Returns
        -------
        list
            All genes numbered. Genes repeated for each of their sgRNAs. 
        
        """
        return ["gene_" + str(i) for i in np.arange(len(self.sgRNAs)) for n in np.arange(self.sgRNAs[i])]
    
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
            a = [np.random.poisson(i, size=1) for i in lambdas]
        elif self.type_dist == "negative binomial":
            a = [np.random.negative_binomial(i, p, size=1) for i in lambdas for p in p_array]
        else:
            raise Exception("Make sure to choose a distribution from those available")
        
        a = np.concatenate(a)
        a = a.astype(float)
        a /= (a.sum())
        a *= self.totals_array[index]
        a = np.round(a)
        
        return a
    
    def _setting_control_libraries(self) -> list:
        """
        Generates values for control libraries.
        
        Returns
        -------
        control : list
            List of arrays, one for each library, generated by the _sum_array() method. 
            
        """
        control = [] 
        
        for i in np.arange(self.num_control):
            control.append(self._sum_array(i, self.lam, self.p))
        
        return control 
    
    def _setting_treatment_libraries(self) -> list:
        """
        Generates values for treatment libraries.
        
        Returns
        -------
        treatment : list
            List of arrays, one for each library, generated by the _sum_array() method. 
            
        """
        treatment = [] 
        
        for i in np.arange(self.num_treatment):
            treatment.append(self._sum_array(-(i+1), self._S_l(), self.p))
        
        return treatment
    
    def _S_l(self) -> np.ndarray:
        """
        Scales the lambdas for treatment libraries. 
        
        Returns
        -------
        np.ndarray 
            Element-wise product of `S` and `lam`.  
            
        """
        return np.multiply(self.S, self.lam)
     
    def _modification(self) -> list:
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
        
        e = ["enriched" for i in np.arange(len(self.g_e)) for n in np.arange(self.g_e[i])]
        d = ["depleted" for i in np.arange(len(self.g_d)) for n in np.arange(self.g_d[i])]
        ntc = ["ntc" for i in np.arange(len(self.g_ntc)) for n in np.arange(self.g_ntc[i])]
        n = ["normal" for i in np.arange(len(self.g_n)) for n in np.arange(self.g_n[i])]
        
        modification = e + d + ntc + n
        
        return modification 
    
    def _sgRNA_df(self) -> pd.DataFrame:
        """
        Puts sgRNAs into a DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Numbered sgRNAs with label "sgRNAs". 
            
        """
        return pd.DataFrame({"sgRNAs": self._sgRNAs()})
    
    def _gene_df(self) -> pd.DataFrame:
        """
        Puts genes into a DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Numbered genes with label "gene". 
            
        """
        return pd.DataFrame({"gene": self._gene()})
    
    def _lam_df(self) -> pd.DataFrame:
        """
        Puts lambda values into a DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Lambda values with label "lambda". 
            
        """
        return pd.DataFrame({"lambda": self.lam})
    
    def _S_df(self) -> pd.DataFrame:
        """
        Puts gene-specific scalar values into a DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Scalar values with label "scalar". 
            
        """
        return pd.DataFrame({"scalar": self.S})
    
    def _S_lam_df(self) -> pd.DataFrame:
        """
        Puts scaled lambda values into a DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Scaled lambda values with label "scaled lambda". 
            
        """
        return pd.DataFrame({"scaled lambda": self._S_l()})
    
    def _controls_df(self) -> pd.DataFrame:
        """
        Puts control libraries into a DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Control libraries with numbered "control_" labels. 
            
        """
        control = {}
        for i in np.arange(self.num_control):
            control["control_" + str(i)] = self._setting_control_libraries()[i]
           
        control = pd.DataFrame(control)
        return control
    
    def _treatments_df(self) -> pd.DataFrame:
        """
        Puts treatment libraries into a DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Treatment libraries with numbered "treatment_" labels. 
            
        """
        treatment = {}
        for i in np.arange(self.num_treatment):
            treatment["treatment_" + str(i)] = self._setting_treatment_libraries()[i]
           
        treatment = pd.DataFrame(treatment)
        return treatment
    
    def _modification_df(self):
        """
        Puts modification assignments into a DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Modification assignments with label "modification". 
            
        """
        return pd.DataFrame({"modification": self._modification()})
            
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
        
        result = pd.concat([
            self._sgRNA_df(), 
            self._gene_df(), 
            self._lam_df(),
            self._S_df(),
            self._S_lam_df(), 
            self._controls_df(), 
            self._treatments_df(), 
            self._modification_df()], 
            axis=1, 
            join="inner")
        
        return result 