import pytest
from screenSim.simulator import Simulator

# some tests, will add more 

trial = Simulator(num_genes = 10)
    
def test_genes():
    assert trial.num_genes == 10

def test_fractions():
    assert trial.fraction_enriched + trial.fraction_depleted + trial.fraction_NTC + trial.fraction_normal == 1
     
def test_totals_array():
    for i in trial.totals_array:
        assert (i < trial.max_total) & (i > trial.min_total)
    
def test_totals_array_2():
    assert len(trial.totals_array) == trial.num_control + trial.num_treatment
    
def test_num_types():
    assert trial.num_e + trial.num_d + trial.num_ntc + trial.num_n == trial.num_genes
    
def test_lambda(): 
    for i in trial.lam:
        assert (i > trial.bounds[0]) & (i < trial.bounds[1])

def test_treatment_sums():
    treatments = trial._setting_treatment_libraries()
    total = trial.totals_array
    for i in range(len(treatments)):
        assert abs((treatments[i].sum() - total[i])/total[i]) < 0.05

def test_control_sums():
    controls = trial._setting_control_libraries()
    total = trial.totals_array
    for i in range(len(controls)):
        assert abs((controls[i].sum() - total[-(i+1)])/total[-(i+1)]) < 0.05
        
        
    
    
    

    
    
    
