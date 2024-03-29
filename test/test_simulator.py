import pytest
import numpy as np
import pandas as pd
from screenSim.simulator import Simulator

@pytest.fixture
def sim() -> Simulator:
    return Simulator(num_genes = 10)
    
def test_genes(sim):
    assert sim.num_genes == 10

def test_sgRNAs(sim):
    sim.num_sgRNAs == sim.num_genes * sim.num_sgRNAs_per_gene

def test_fractions(sim):
    assert sim.fraction_enriched + sim.fraction_depleted + sim.fraction_NTC + sim.fraction_normal == 1
     
def test_totals_array(sim):
    for i in sim.totals_array:
        assert (i < sim.max_total) & (i > sim.min_total)
    
def test_totals_array_len(sim):
    assert len(sim.totals_array) == sim.num_control + sim.num_treatment
        
def test_number_non_targeting(sim):
    df = sim.sample()
    assert df.gene.str.contains("non-targeting").sum() == sim.num_ntc == round(sim.num_genes * sim.fraction_NTC) * sim.num_sgRNAs_per_gene
    
def test_number_enriched(sim):
    df = sim.sample()
    assert (df.modification == "enriched").sum() == sim.num_e == round(sim.num_genes * sim.fraction_enriched) * sim.num_sgRNAs_per_gene
    
def test_number_depleted(sim):
    df = sim.sample()
    assert (df.modification == "depleted").sum() == sim.num_d == round(sim.num_genes * sim.fraction_depleted) * sim.num_sgRNAs_per_gene
    
def test_number_ntcs(sim):
    df = sim.sample()
    assert (df.modification == "ntc").sum() == sim.num_ntc == round(sim.num_genes * sim.fraction_NTC) * sim.num_sgRNAs_per_gene

def test_control_sums(sim):
    controls = pd.DataFrame()
    sim._setting_control_libraries(controls)
    total = sim.totals_array
    for i in range(sim.num_control):
        assert abs((controls[controls.columns[i]].sum() - total[i])/total[i]) < 0.05

def test_treatment_sums(sim):
    treatments = pd.DataFrame()
    sim._setting_treatment_libraries(treatments)
    total = sim.totals_array
    for i in range(sim.num_treatment):
        assert abs((treatments[treatments.columns[i]].sum() - total[-(i+1)])/total[-(i+1)]) < 0.05