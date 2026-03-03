/**
* Name: main
* Setup basic experiment 
* Author: Cheikhou Akhmed KANE
* Tags: 
*/


model main

/* Insert your model definition here */

import "EcoSysMLStructure.gaml"
import "EcoSysMLBehavior.gaml"

global {
    // --- State Variables (Units: SOC in g/kg | Yield in t/ha) ---
    // These are initialized by the parameters in the experiment block
    
    // Average Ecological Impact of actions
    float mu_delta_soc_fallow;
    float mu_delta_soc_manure;
    float mu_delta_soc_tree;
    float mu_delta_soc_baseline;
    
    // Average Economic Impact of actions
    float mu_yield_fallow;
    float mu_yield_manure;
    float mu_yield_tree;
    float mu_yield_baseline;
    
    // Variance
    float sigma_soc <- 0.1;
    float sigma_yield <- 0.5;
    
    // Mapping lists for Batch Decision Making
    list<float> soc_effects;
    list<float> yield_effects;

    init {
        // Sync parameters with lists for the execute_on action
        soc_effects <- [mu_delta_soc_fallow, mu_delta_soc_manure, mu_delta_soc_tree, mu_delta_soc_baseline];
        yield_effects <- [mu_yield_fallow, mu_yield_manure, mu_yield_tree, mu_yield_baseline];
        
        create Actor {
            id <- "Farmer_1";
            ownedResources <- list(Parcel);
        }
    }
}

experiment "AgroCarbonSimulation" type: gui {
    // --- Ecological Parameters Category ---
    parameter "SOC Effect (Fallow): " var: mu_delta_soc_fallow min: -5.0 max: 5.0 category: "Ecological" ;
    parameter "SOC Effect (Manure): " var: mu_delta_soc_manure min: -5.0 max: 5.0 category: "Ecological" ;
    parameter "SOC Effect (Tree): " var: mu_delta_soc_tree min: -5.0 max: 5.0 category: "Ecological" ;
    parameter "SOC Effect (Baseline): " var: mu_delta_soc_baseline min: -5.0 max: 5.0 category: "Ecological" ;
    
    // --- Economic Parameters Category ---
    parameter "Yield (Fallow): " var: mu_yield_fallow min: 0.0 max: 20.0 category: "Economic" ;
    parameter "Yield (Manure): " var: mu_yield_manure min: 0.0 max: 20.0 category: "Economic" ;
    parameter "Yield (Tree): " var: mu_yield_tree min: 0.0 max: 20.0 category: "Economic" ;
    parameter "Yield (Baseline): " var: mu_yield_baseline min: 0.0 max: 20.0 category: "Economic" ;
    
    //--- Uncertainty (Sigma) ---
    parameter "SOC Noise (sigma): " var: sigma_soc min: 0.0 max: 2.0 category: "Uncertainty" ;
    parameter "Yield Noise (sigma): " var: sigma_yield min: 0.0 max: 5.0 category: "Uncertainty" ;

    output {
        display "Territory View" {
            grid Parcel border: #black;
            species Tree aspect: base;
        }
    }
}