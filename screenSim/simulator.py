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
        fraction_enriched = 0.2,
        fraction_depleted = 0.2,
        fraction_NTC = 0.2):

        self.totals_array = np.random.randint(self.min_total, self.max_total, size = self.num_treatment + self.num_control)

        self.bounds = [10, 30]

        self.sgRNAs = self._num_sgRNAs()

        self.num_e = round(len(self.sgRNAs) * self.fraction_enriched)
        self.num_d = round(len(self.sgRNAs) * self.fraction_depleted)
        self.num_ntc = round(len(self.sgRNAs) * self.fraction_NTC)
        self.num_n = round(len(self.sgRNAs) * self.fraction_normal)

        self.lam = [np.random.uniform(self.bounds[0], self.bounds[1]) for gene in self.sgRNAs for num in np.arange(gene)]

        #make option of whether nbinomial or poisson, trucated normal 

    def _sgRNAs(self):
        return ["sgRNA_" + str(int(i)) for i in np.arange(self.sgRNAs.sum())]

    def _gene(self):
        
        return ["gene_" + str(i) for i in np.arange(len(self.sgRNAs)) for n in np.arange(self.sgRNAs[i])]


    def _num_sgRNAs(self):
        sgRNAs = np.random.normal(loc=self.avg_num_sgRNAs, scale=1, size=self.num_genes)
        sgRNAs = np.round(sgRNAs)
        return sgRNAs 


    def _sum_array(self, index, lambdas):

        a = [np.random.poisson(i, size=1) for i in lambdas]
        a = np.concatenate(a)
        a = a.astype(float)
        a /= (a.sum())
        a *= self.totals_array[index]
        a = np.round(a)

        return a

    def _setting_treatment_libraries(self):
        treatment = [] 

        for i in np.arange(self.num_treatment):
            treatment.append(self._sum_array(i, self._S_l()))

        return treatment

    def _setting_control_libraries(self):
        control = [] 

        for i in np.arange(self.num_control):
            control.append(self._sum_array(-(i+1), self.lam))

        return control


    def _g_e(self):
        return self.sgRNAs[0: self.num_e]      

    def _g_d(self):
        return self.sgRNAs[self.num_e: self.num_e + self.num_d]

    def _g_ntc(self):
        return self.sgRNAs[self.num_e + self.num_d: self.num_e + self.num_d + self.num_ntc]

    def _g_n(self):
        return self.sgRNAs[self.num_e + self.num_d + self.num_ntc: self.num_e + self.num_d + self.num_ntc + self.num_n]

    def _S(self):

        # currently different each time called, the S_l in dataframe doens't reflect the one used for treatments,
        # add in constructor?

        S = []

        g_e = self._g_e()
        g_d = self._g_d()
        g_ntc = self._g_ntc()
        g_n = self._g_n()

        for i in g_e:
            g_scalar = np.random.uniform(1.2, 2.0)
            for n in np.arange(i):
                S.append(g_scalar)

        for i in g_d:
            g_scalar = np.random.uniform(0.2, 1.0)
            for n in np.arange(i):
                S.append(g_scalar)

        for i in g_ntc:
            for n in np.arange(i):
                S.append(1)

        for i in g_n:
            for n in np.arange(i):
                S.append(1)

        return S 

    def _S_l(self):
        return np.multiply(self._S(), self.lam)



    def _type_of_change(self):

        type_of_change = []

        g_e = self._g_e()
        g_d = self._g_d()
        g_ntc = self._g_ntc()
        g_n = self._g_n()

        e = ["enriched" for i in np.arange(len(g_e)) for n in np.arange(g_e[i])]
        d = ["depleted" for i in np.arange(len(g_d)) for n in np.arange(g_d[i])]
        ntc = ["ntc" for i in np.arange(len(g_ntc)) for n in np.arange(g_ntc[i])]
        n = ["normal" for i in np.arange(len(g_n)) for n in np.arange(g_n[i])]

        type_of_change = e + d + ntc + n

        return type_of_change 


    def sample(self):
    

        # reorganize this to make code clearer
        sgRNA = pd.DataFrame({"sgRNAs": self._sgRNAs()})
        gene = pd.DataFrame({"gene": self._gene()})
        lam = pd.DataFrame({"lambda": self.lam})
        S_lam = pd.DataFrame({"modified lambda": self._S_l()})
        control = pd.DataFrame(self._setting_control_libraries()).T
        treatment = pd.DataFrame(self._setting_treatment_libraries()).T
        type_of_change = pd.DataFrame({"type": self._type_of_change()})

        result = pd.concat([sgRNA, gene, lam, S_lam, control, treatment, type_of_change], axis=1, join="inner")

        return result 