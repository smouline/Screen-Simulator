import pytest
import numpy as np
import pandas as pd
from screenSim.simulator import Simulator


@pytest.fixture
def df() -> Simulator:
    return Simulator(num_genes = 10)
    
def test_genes(df):
    assert df.num_genes == 10

def test_fractions(df):
    assert df.fraction_enriched + df.fraction_depleted + df.fraction_NTC + df.fraction_normal == 1
     
def test_totals_array(df):
    for i in df.totals_array:
        assert (i < df.max_total) & (i > df.min_total)
    
def test_totals_array_2(df):
    assert len(df.totals_array) == df.num_control + df.num_treatment
    
def test_num_types(df):
    assert len(df.g_e) + len(df.g_d) + len(df.g_ntc) + len(df.g_n) == df.num_genes
    
def test_lambda(df): 
    for i in df.lam:
        assert (i > df.bounds[0]) & (i < df.bounds[1])

def test_control_sums(df):
    controls = df._setting_control_libraries()
    total = df.totals_array
    for i in range(len(controls)):
        assert abs((controls[i].sum() - total[i])/total[i]) < 0.05

def test_treatment_sums(df):
    treatments = df._setting_treatment_libraries()
    total = df.totals_array
    for i in range(len(treatments)):
        assert abs((treatments[i].sum() - total[-(i+1)])/total[-(i+1)]) < 0.05
        
def test_sgRNA_average(df):
    assert (np.mean(df.sgRNAs) - df.avg_num_sgRNAs)/df.avg_num_sgRNAs < 0.10
    

        
        
        
        
    
    
    

    
    
    
