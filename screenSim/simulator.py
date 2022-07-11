import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Simulator:
    def __init__(
        self, 
        num_genes = 20000, 
        avg_num_sgRNAs = 5, 
        num_treatment = 2, 
        num_control = 2, 
        min_total = 1000,
        max_total = 100000,
        total_NTCs = 1000,
        fraction_enriched = 0.1,
        fraction_depleted = 0.1,
        fraction_NTC = 0.1):
        
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
            
        Raises
        ------
        Exception
            If the total of fractions (enriched, depleted, NTC) exceeds 1. 
        
        """
        self.num_genes = num_genes
        self.avg_num_sgRNAs = avg_num_sgRNAs
        self.num_treatment = num_treatment
        self.num_control = num_control
        self.min_total = min_total
        self.max_total = max_total
        self.total_NTCs = total_NTCs
        
        total_fractions = fraction_enriched + fraction_depleted + fraction_NTC
        
        if ((total_fractions > 0.0) & (total_fractions <= 1.0)):
            self.fraction_enriched = fraction_enriched
            self.fraction_depleted = fraction_depleted
            self.fraction_NTC = fraction_NTC
            self.fraction_normal = 1.0 - (fraction_enriched + fraction_depleted + fraction_NTC)
        else:
            raise Exception("Fractions total cannot exceed 1.")
        
        self.totals_array = np.random.randint(self.min_total, self.max_total, size = self.num_treatment + self.num_control)
        # print(self.totals_array)
        
        self.normal_bounds = np.array([10, 30])
    
    def _gene(self):
        """
        Generates list of numbered genes for use in sample() DataFrame. 
        
        Returns
        -------
        list
            The elements of the list are in numerical order up to the 
            number of genes in `num_genes`.
        
        """
        return ["gene_" + str(i) for i in np.arange(self.num_genes)]
    
    
    def _num_sgRNAs(self):
        """
        Generates a number of sgRNAs per gene. 
        
        Returns
        -------
        sgRNAs : array
            The values of the array follow a normal distrubution with the
            mean being `avg_num_sgRNAs`.
        
        """
        sgRNAs = np.random.normal(loc=self.avg_num_sgRNAs, scale=1, size=self.num_genes)
        sgRNAs = np.round(sgRNAs)
        return sgRNAs 
    
    def _scale_e(self):
        """
        Scales the no effect bounds up. 
        
        Returns
        -------
        array
            The scaling factor is a float within the range [1.2, 2.0)
        
        """
        return self.normal_bounds * np.random.uniform(1.2, 2.0)
    
    def _scale_d(self):
        """
        Scales the no effect bounds down. 
        
        Returns
        -------
        array
            The scaling factor is a float within the range [0.2, 1.0)
        
        """
        return self.normal_bounds * np.random.uniform(0.2, 1.0)
    
    def _lamda(self, bounds):
        """
        Generates value for lamba for poisson distrubution. 
        
        Parameters
        ----------
        bounds : array
            Array with 2 elements--min and max for lambda
        
        Returns
        -------
        float 
            The lamda float is within the the first (included) and second element of `bounds`
        
        """
        return np.random.uniform(bounds[0], bounds[1])
    
    def _e_dist(self):
        """
        Generates poisson distrubution for enriched genes. 
        
        Returns
        -------
        array
            description
        
        """
        return np.random.poisson(self._lamda(self._scale_e()), round(self.num_genes * self.fraction_enriched))
    
    def _d_dist(self):
        """
        Generates poisson distrubution for depleted genes. 
        
        Returns
        -------
        array
            description
        
        """
        return np.random.poisson(self._lamda(self._scale_d()), round(self.num_genes * self.fraction_depleted))
    
    def _ntc_dist(self):
        """
        Generates poisson distrubution for ntc genes. 
        
        Returns
        -------
        array
            description
        
        """
        return np.random.poisson(self._lamda(self.normal_bounds), round(self.num_genes * self.fraction_NTC))
    
    def _ne_dist(self):
        """
        Generates poisson distrubution for normal genes. 
        
        Returns
        -------
        array
            description
        
        """
        return np.random.poisson(self._lamda(self.normal_bounds), round(self.num_genes * self.fraction_normal))
        
    def _sum_array(self, index):
        """
        Creates an array of random integers with a specified sum.
        
        Parameters
        ----------
        index : int
            The index to specify which total to use from `totals_array` 
            defined in the constructor. 
            
        Returns
        -------
        a : array
            array of randomly generated integers with sum of element from `totals_array`    
        
        """
        a = np.concatenate((self._e_dist(), self._d_dist(), self._ntc_dist(), self._ne_dist()))
        a = a.astype(float)
        a /= (a.sum())
        a *= self.totals_array[index]
        a = np.round(a)
        # print(a)
        # print(a.sum())
        return a
    
    def _setting_treatment_libraries(self):
        """
        Generates values for treatment libraries.
        
        Returns
        -------
        treatment : list
            `treatment` is a list of arrays, one for each library, 
            generated by the _sum_array() method. 
            
        """
        treatment = [] 
        
        for i in np.arange(self.num_treatment):
            treatment.append(self._sum_array(i))
        
        return treatment
    
    def _setting_control_libraries(self):
        """
        Generates values for control libraries.
        
        Returns
        -------
        control : list
            `control` is a list of arrays, one for each library, 
            generated by the _sum_array() method. 
            
        """
        control = [] 
        
        for i in np.arange(self.num_control):
            control.append(self._sum_array(-(i+1)))
        
        return control
        
    def _type_of_change(self):
        """
        Sets number of enriched, depleted, NTC, and normal genes. 
        
        Returns
        -------
        type_of_change : list
            The list is filled with strings of enriched, depleted, NTC, 
            and normal for a gene. The number of times a type is in the 
            list is based on the fractional representation specified upon 
            initialization.
            
        """
        type_of_change = ["enriched"] * round(self.num_genes * self.fraction_enriched)
        type_of_change += ["depleted"] * round(self.num_genes * self.fraction_depleted)
        type_of_change += ["NTC"] * round(self.num_genes * self.fraction_NTC)
        type_of_change += ["normal"] * round(self.num_genes * self.fraction_normal)
        return type_of_change 
    
    
    def sample(self):
        """
        Generates DataFrame with observations for the simulation. 
        
        Returns
        -------
        result : DataFrame 
            This DataFrame concatenates information from the _gene(),
            _num_sgRNAs(), _setting_treatment_libraries(),
            _setting_control_libraries(), _type_of_change() methods.  
            
        """
        gene = pd.DataFrame({"gene": self._gene()})
        sgRNAs = pd.DataFrame({"sgRNAs": self._num_sgRNAs()})
        treatment = pd.DataFrame(self._setting_treatment_libraries()).T
        control = pd.DataFrame(self._setting_control_libraries()).T
        type_of_change = pd.DataFrame({"type": self._type_of_change()})
        
        result = pd.concat([gene, sgRNAs, treatment, control, type_of_change], axis=1, join="inner")

        return result 