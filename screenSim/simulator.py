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
        min_total: int = 1000,
        max_total: int = 100000,
        total_NTCs: int = 1000,
        fraction_enriched: float = 0.2,
        fraction_depleted: float = 0.2,
        fraction_NTC: float = 0.2,
        type_dist: int = 2):

        self.num_genes = num_genes
        self.avg_num_sgRNAs = avg_num_sgRNAs
        self.num_control = num_control
        self.num_treatment = num_treatment
        self.min_total = min_total
        self.max_total = max_total
        self.total_NTCs = total_NTCs
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
        self.totals_array = np.random.randint(self.min_total, self.max_total, size = self.num_treatment + self.num_control)
     

    def _init_fractions(self, e: float, d: float, ntc: float):
        total = e + d + ntc

        if ((total > 0.0) & (total <= 1.0)):
            self.fraction_enriched = e
            self.fraction_depleted = d
            self.fraction_NTC = ntc
            self.fraction_normal = 1.0 - (e + d + ntc)
        else:
            raise Exception("Fractions total cannot exceed 1.") 


    def _init_num_sgRNAs(self):
        sgRNAs = np.random.normal(loc=self.avg_num_sgRNAs, scale=1, size=self.num_genes)
        sgRNAs = np.round(sgRNAs)
        self.sgRNAs = sgRNAs 


    def _split_genes(self):
        num_e = round(len(self.sgRNAs) * self.fraction_enriched)
        num_d = round(len(self.sgRNAs) * self.fraction_depleted)
        num_ntc = round(len(self.sgRNAs) * self.fraction_NTC)
        num_n = round(len(self.sgRNAs) * self.fraction_normal)

        self.g_e = self.sgRNAs[0: num_e]
        self.g_d = self.sgRNAs[num_e: num_e + num_d]
        self.g_ntc = self.sgRNAs[num_e + num_d: num_e + num_d + num_ntc]
        self.g_n = self.sgRNAs[num_e + num_d + num_ntc: num_e + num_d + num_ntc + num_n]


    def _init_lambda(self):
        self.lam = np.random.uniform(self.bounds[0], self.bounds[1], size = int(self.sgRNAs.sum()))


    def _init_p(self):
        self.p = np.random.random(size = int(self.sgRNAs.sum()))


    def _init_S(self):
        S = []

        for i in self.g_e:
            g_scalar = np.random.uniform(1.2, 2.0)
            for n in np.arange(i):
                S.append(g_scalar)

        for i in self.g_d:
            g_scalar = np.random.uniform(0.2, 1.0)
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
        return ["sgRNA_" + str(int(i)) for i in np.arange(self.sgRNAs.sum())]


    def _gene(self) -> list:
        return ["gene_" + str(i) for i in np.arange(len(self.sgRNAs)) for n in np.arange(self.sgRNAs[i])]


    def _sum_array(self, index: int, lambdas: np.ndarray, p_array: np.ndarray) -> np.ndarray:

        if self.type_dist == 1:
            a = [np.random.poisson(i, size=1) for i in lambdas]
        elif self.type_dist == 2:
            a = [np.random.negative_binomial(i, p, size=1) for i in lambdas for p in p_array]
        else:
            raise Exception("Make sure to choose a distrubtion type from the available ints")

        a = np.concatenate(a)
        a = a.astype(float)
        a /= (a.sum())
        a *= self.totals_array[index]
        a = np.round(a)

        return a


    def _setting_treatment_libraries(self) -> list:
        treatment = [] 

        for i in np.arange(self.num_treatment):
            treatment.append(self._sum_array(i, self._S_l(), self.p))

        return treatment


    def _setting_control_libraries(self) -> list:
        control = [] 

        for i in np.arange(self.num_control):
            control.append(self._sum_array(-(i+1), self.lam, self.p))

        return control


    def _S_l(self) -> np.ndarray:
        return np.multiply(self.S, self.lam)


    def _type_of_change(self) -> list:

        type_of_change = []

        e = ["enriched" for i in np.arange(len(self.g_e)) for n in np.arange(self.g_e[i])]
        d = ["depleted" for i in np.arange(len(self.g_d)) for n in np.arange(self.g_d[i])]
        ntc = ["ntc" for i in np.arange(len(self.g_ntc)) for n in np.arange(self.g_ntc[i])]
        n = ["normal" for i in np.arange(len(self.g_n)) for n in np.arange(self.g_n[i])]

        type_of_change = e + d + ntc + n

        return type_of_change 


    def sample(self, seed: int = 10) -> pd.DataFrame:

        # reorganize this to make code clearer

        np.random.seed(seed)

        sgRNA = pd.DataFrame({"sgRNAs": self._sgRNAs()})
        gene = pd.DataFrame({"gene": self._gene()})
        lam = pd.DataFrame({"lambda": self.lam})
        S_lam = pd.DataFrame({"scaled lambda": self._S_l()})
        control = pd.DataFrame(self._setting_control_libraries()).T
        treatment = pd.DataFrame(self._setting_treatment_libraries()).T
        type_of_change = pd.DataFrame({"type": self._type_of_change()})

        result = pd.concat([sgRNA, gene, lam, S_lam, control, treatment, type_of_change], axis=1, join="inner")
        
        return result 