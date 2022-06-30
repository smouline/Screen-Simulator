import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Simulator:
    def __init__(
        self, 
        num_genes = 200000, 
        avg_num_sgRNAs = 5, 
        num_treatment = 2, 
        num_control = 2, 
        min_total = 1000,
        max_total = 100000,
        total_NTCs = 1000,
        fraction_enriched = 0.1,
        fraction_depleted = 0.1,
        fraction_NTC = 0.1):
        
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
    
    def _gene(self):
        return ["gene_" + str(i) for i in np.arange(self.num_genes)]
    
    
    def _num_sgRNAs(self):
        sgRNAs = np.random.normal(loc=5, scale=1, size=self.num_genes)
        sgRNAs = np.round(sgRNAs)
        return sgRNAs 
        
    def _sum_array(self, index):
        a = np.random.random(self.num_genes)
        a /= a.sum()
        a *= self.totals_array[index]
        a = np.round(a)
        return a
    
    def _setting_treatment_libraries(self):
        treatment = [] 
        
        for i in np.arange(self.num_treatment):
            treatment.append(self._sum_array(i))
        
        return treatment
    
    def _setting_control_libraries(self):
        control = [] 
        
        for i in np.arange(self.num_control):
            control.append(self._sum_array(-i))
        
        return control
        
    def _type_of_change(self):        
        type_of_change = ["enriched"] * round(self.num_genes * self.fraction_enriched)
        type_of_change += ["depleted"] * round(self.num_genes * self.fraction_depleted)
        type_of_change += ["NTC"] * round(self.num_genes * self.fraction_NTC)
        type_of_change += ["normal"] * round(self.num_genes * self.fraction_normal)
        return type_of_change 
    
    
    def sample(self):
        
        gene = pd.DataFrame({"gene": self._gene()})
        sgRNAs = pd.DataFrame({"sgRNAs": self._num_sgRNAs()})
        treatment = pd.DataFrame(self._setting_treatment_libraries()).T
        control = pd.DataFrame(self._setting_control_libraries()).T
        type_of_change = pd.DataFrame({"type": self._type_of_change()})
        
        result = pd.concat([gene, sgRNAs, treatment, control, type_of_change], axis=1, join="inner")

        return result 