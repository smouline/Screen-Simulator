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
        n_prior: float = 1.4,
        p_prior: float = 2.5e-3,
        num_bins: int = 50,
        e_scalar_min: float = 0.5,
        e_scalar_max: float = 2.0,
        d_scalar_min: float = -2.0,
        d_scalar_max: float = -0.5,
        type_dist: str = "negative binomial",
        seed: int = 42):
        
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
        n_prior : float
            The n param for the overall negative binomial.
        p_prior : float
            The p param for the overall negative binomial. 
        num_bins : int
            The number of bins to split the overall negative binomial distribution into (see _init_n() for more info). 
        e_scalar_min : float
            The lower bound for the n enriched scalars. 
        e_scalar_max : float
            The upper bound for the n enriched scalars.
        d_scalar_min : float
            The lower bound for the n depleted scalars. 
        d_scalar_max : float
            The upper bound for the n depleted scalars.
        type_dist : str
            Either "poisson" or "negative binomial" distribution. Note: lambda in poisson will be referred to as n here. 
        seed : int
            Simulators are repeatable for the same `seed`.
        
        """ 
        np.random.seed(seed)
        
        self.num_genes = int(num_genes)
        self.num_sgRNAs_per_gene = int(num_sgRNAs_per_gene)
        self.num_control = int(num_control)
        self.num_treatment = int(num_treatment)
        self.type_dist = type_dist
        self.n_prior = n_prior
        self.p_prior = p_prior
        self.num_bins = int(num_bins)
        
        self._init_fractions(fraction_enriched, fraction_depleted, fraction_NTC)
        self._init_totals_bounds(int(min_total), int(max_total))
        self._init_e_bounds(e_scalar_min, e_scalar_max)
        self._init_d_bounds(d_scalar_min, d_scalar_max)
        
        self._num_sgRNAs()
        self._init_count_totals()
        self._split_genes()
        self._split_sgRNAs()
        self._init_sgRNA()
        self._init_gene()
        self._init_n()
        self._init_p()
        self._init_S()
        self._init_viability()
        self._add_S_noise()
        self._mult_S_n()
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
            
    def _init_e_bounds(self, lower: float, upper: float):
        """
        Initializes the enriched scalar bounds for lambda.
        
        Raises
        ------
        Exception
            If lower/upper bounds <= 0 and/or lower > upper.
        
        """
        if (lower < upper):
            self.e_scalar_min = lower
            self.e_scalar_max = upper
        else:
            raise Exception("e_scalar_min should be less than e_scalar_max.")
     
    def _init_d_bounds(self, lower: float, upper: float):
        """
        Initializes the depleted scalar bounds for lambda.
        
        Raises
        ------
        Exception
            If lower/upper bounds <= 0 and/or lower > upper.
        
        """
        if (lower < upper):
            self.d_scalar_min = lower
            self.d_scalar_max = upper
        else:
            raise Exception("d_scalar_min should be less than d_scalar_max.")
   
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
        self.num_g_e = round(self.num_genes * self.fraction_enriched)
        self.num_g_d = round(self.num_genes * self.fraction_depleted)
        self.num_g_ntc = round(self.num_genes * self.fraction_NTC)
        
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
    
    def _init_n(self):
        """
        Initializes an n for each sgRNA. 
        Splits the overall negative binomial into bins and assigns n based on those bins. 
        This way, sampling is very similar to the overall negative binomial and not overdispersed.
        
        """
        n = np.zeros(shape = self.num_sgRNAs)
        
        overall = np.random.negative_binomial(n = self.n_prior, p = self.p_prior, size = self.num_sgRNAs)
        
        hist = np.histogram(overall, bins = self.num_bins)
        
        bins = hist[1]
            
        for i in np.arange(self.num_bins):
            bin_left, bin_right = bins[i], bins[i+1]
            mask = (overall >= bin_left) & (overall <= bin_right)
            n[mask] = (bin_left + bin_right)/2
    
        self.n = n

    def _init_p(self):
        """
        Initializes a probability for each sgRNA.
        
        """
        self.p = np.full(shape = self.num_sgRNAs, fill_value = 0.9)
    
    def _init_S(self):
        """
        Initializes gene-specific n scalars for each gene depending on the gene classification. 
        
        """
        S = np.zeros(self.num_sgRNAs)
        
        gene_e_scalars = np.random.uniform(self.e_scalar_min, self.e_scalar_max, size = self.num_g_e)
        gene_d_scalars = np.random.uniform(self.d_scalar_min, self.d_scalar_max, size = self.num_g_d)
        
        S[:self.num_e] = np.repeat(gene_e_scalars, self.num_sgRNAs_per_gene)
        S[self.num_e: self.num_e + self.num_d] = np.repeat(gene_d_scalars, self.num_sgRNAs_per_gene)
        
        self.S = np.exp2(S)
        self.S_pre_noise = S
        
    def _init_viability(self):
        """
        """
        v = np.random.beta(a = 5, b = 1, size = self.num_genes)
        self.v = np.repeat(v, self.num_sgRNAs_per_gene)
        
    def _add_S_noise(self):
        """
        """
        noise_gene = np.array([np.random.beta(5, 1, size = self.num_sgRNAs_per_gene) for i in np.arange(self.num_genes)])
        noise_sg = np.reshape(noise_gene, newshape = self.num_sgRNAs)
        self.noise = noise_sg
        
        self.S_post_noise = self.S * self.noise  
        
    def _mult_S_n(self):
        """
        Scales n for treatment libraries by performing an element-wise product of `self.S` and `self.n`.
            
        """
        self.S_n = self.S_post_noise * self.n * self.v
     
    def _init_modification(self):
        """
        Labels each sgRNA as enriched, depleted, ntc, or normal. 
        
        """
        
        mod = ["normal"] * self.num_sgRNAs
        
        mod[:self.num_e] = ["enriched"] * self.num_e
        mod[self.num_e: self.num_e + self.num_d] = ["depleted"] * self.num_d
        mod[self.num_e + self.num_d: self.num_e + self.num_d + self.num_ntc] = ["ntc"] * self.num_ntc
        
        self.modification = mod
        
    def _sampling(self, n_array: np.ndarray, p_array: np.ndarray) -> np.ndarray:
        """
        Generates count values for one library using each sgRNA's n/(p) value(s). 
        
        Parameters
        ----------
        n_array: np.ndarray
            To use as n in negative binomial or lam in in poisson.
        p_array: np.ndarray 
            Probabilities to use as p in negative binomial.
            
        Raises
        ------
        Exception
            If input is not "poisson" or "negative binomial".
            
        Returns
        -------
        counts : np.ndarray
            Count values for a given library. 
        
        """
        if self.type_dist == "poisson":
            counts = np.random.poisson(n_array)
        elif self.type_dist == "negative binomial":
            counts = np.random.negative_binomial(n_array, p_array)
        else:
            raise Exception("Make sure to choose a distribution from those available.")
            
        return counts
        
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
            Array with a total of an element from `totals_array`    
        
        """
        norm = norm.astype(float)
        norm /= (norm.sum())
        norm *= self.totals_array[index]
        norm = np.round(norm)
      
        return norm
    
    def _setting_control_libraries(self, df: pd.DataFrame):
        """
        Generates values for control libraries with _sum_array() and appends each library as a column to `df`.
        
        Parameters
        ----------
        df: pd.DataFrame
            DataFrame to add control libraries to as columns. 
            
        """
        for i in np.arange(self.num_control):
            df[f"control_{i}"] = self._normalize(self._sampling(self.n, self.p), i)

    def _setting_treatment_libraries(self, df: pd.DataFrame):
        """
        Generates values for treatment libraries with _sum_array() and appends each library as a column to `df`.
        
        Parameters
        ----------
        df: pd.DataFrame
            DataFrame to add treatment libraries to as columns. 
            
        """
        for i in np.arange(self.num_treatment):
            df[f"treatment_{i}"] = self._normalize(self._sampling(self.S_n, self.p), -(i+1))
            
    def _norm(self, array: np.ndarray) -> np.ndarray:
        """
        Normalizes array to sum 1. 
        
        Parameters
        ----------
        array: np.ndarray
            Array to normalize. 
        
        Returns
        -------
        array : np.ndarray
            Array with a total of 1.
            
        """
        array = array / array.sum()
        return array
         
    def _log_fold_change(self, df: pd.DataFrame):
        """
        Calculates log2 fold change between treatment and control libraries. 
        
        Parameters
        ----------
        df: pd.DataFrame
            DataFrame to add log fold change data to in a column. 
            
        """
        norm_controls = [self._norm(df[f"control_{i}"].values) for i in np.arange(self.num_control)]
        norm_treatments = [self._norm(df[f"treatment_{i}"].values) for i in np.arange(self.num_treatment)]
        
        c_mean = sum(norm_controls)/len(norm_controls)
        t_mean = sum(norm_treatments)/len(norm_treatments)
        
        df["control_mean"] = c_mean
        df["treatment_mean"] = t_mean
        
        with np.errstate(divide="ignore", invalid="ignore"):
            log2fc = np.ma.log2(t_mean/c_mean)
            log2fc[log2fc.mask] = np.nan
            df["lfc"] = log2fc.data
        
    def sample(self) -> pd.DataFrame:
        """
        Generates DataFrame with observations for the simulation. 
        
        Returns
        -------
        result : pd.DataFrame 
            columns: sgRNA, gene, n, n_scalar, scaled_n, modification, 
            each control and treatment library as a column, and the log fold change (lfc)
            
        """  
        result = pd.DataFrame({
                "sgRNA": self.sgRNA, 
                "gene": self.gene, 
                "n": self.n,
                "n_scalar": self.S,
                "viability": self.v,
                "noise": self.noise,
                "scaled_n": self.S_n,
                "modification": self.modification
            })

        self._setting_control_libraries(result)
        self._setting_treatment_libraries(result)
        self._log_fold_change(result)
        
        return result
    
    def ma_plot(self): 
        """
        Plots an MA plot with the log2 fold change and the mean of the control libraries for each gene modification. 
        
        """
        sim = self.sample()
        
        plt.figure(figsize=(5,5), dpi=100)
        
        for m in sim.modification.unique():
            plt.scatter(
                sim[sim.modification == m].control_mean,
                sim[sim.modification == m].lfc,
                label=m,
                alpha=0.05)
        
        plt.axhline(0, linestyle="dashed", color="black")
        plt.legend()
        plt.xscale("log")
        plt.xlabel("Control Mean")
        plt.ylabel("Log2 Fold Change")
        plt.show()
        
    def correlation_plot(self):
        """
        Plots a correlation scatter plot with the log of control mean on the x-axis 
        and the log of treatment mean on the y-axis for each gene modification. 
        
        """
        sim = self.sample()
        
        plt.figure(figsize=(5,5), dpi=100)
        
        for m in sim.modification.unique():
            plt.scatter(
                sim[sim.modification == m].control_mean,
                sim[sim.modification == m].treatment_mean,
                label=m,
                alpha=0.05)
        
        plt.plot(
            np.logspace(-7, -2),
            np.logspace(-7, -2),
            color="black",
            linestyle="dashed")
        
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Control Mean")
        plt.ylabel("Treatment Mean")
        plt.show()
        
    def lfc_plot(self):
        """
        Plots the fraction the density of sgRNAs against the log2 fold change for each gene modification.   
        
        """
        sim = self.sample()
        
        plt.figure(figsize=(5,3), dpi=150)

        for m in sim.modification.unique():
            plt.hist(
                sim[sim.modification == m].lfc.values, 
                bins=200, 
                density=True, 
                alpha=0.5,
                label=m)

        plt.ylabel("Fraction of sgRNAs")
        plt.xlabel("Log2 Fold Change")
        plt.legend()
        plt.show()