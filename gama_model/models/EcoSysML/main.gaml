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
	// --- File Loading ---
    // Loads the 8 soil signatures from the CSV file
    csv_file soil_ref_file <- csv_file("../../includes/testTerritory/core/soils.csv", true);
    
    // --- Environment Dimensions ---
    int H <- 10;
    int W <- 10;
    
    // --- Context Specs ---
    // S: Number of soil types, M: Tree maturity age
    int S <- 8;
    int M <- 7; 
    
    // --- Policy Parameters ---
    float alpha <- 0.5; // Trade-off between environmental and economic rewards
    
    // --- State Variables (Units: SOC in g/kg | Yield in t/ha) ---
    // These are initialized by the parameters in the experiment block
    
    // Average Ecological Impact of actions
    float mu_delta_soc_fallow <- 0.5;
    float mu_delta_soc_manure <- 0.8;
    float mu_delta_soc_tree <- 1.0;
    float mu_delta_soc_baseline <- 0.2;
    
    // Average Economic Impact of actions
    float mu_yield_fallow <- 0.0;
    float mu_yield_manure <- 0.0;
    float mu_yield_tree <- 0.0;
    float mu_yield_baseline <- 1.5;
    
    // Variance
    float sigma_soc <- 0.14;
    float sigma_yield <- 0.5;
    
    // Mapping lists for Batch Decision Making
    list<float> soc_effects;
    list<float> yield_effects;
    
    // Reference to the biophysical engine agent
    Task biophysical_engine;
    
    GymAgent gym_interface;

    init {
    	
		// Instantiate Gym Agent
		create GymAgent number: 1 returns: gymagents;
		gym_interface <- first(gymagents);
		
		ask gym_interface {
		    // Initialisation des matrices de SOC et de Rendement (10x10)
		    matrix soc_init <- 0.0 as_matrix({10,10});
		    matrix yield_init <- 0.0 as_matrix({10,10});
		    
		    // On remplit la map data pour le premier reset de Python
		    data <- [
		        "State":: 0 as_matrix({10,10}), // État initial (ex: types de cultures)
		        "Reward":: 0.0,
		        "Terminated":: false,
		        "Truncated":: false,
		        "Info":: [
		            "soc_map":: soc_init,
		            "yield_map":: yield_init
		        ]
		    ];
		}
		
		write "GAMA: GymAgent interface initialized and ready.";
        
    	// --- 1. Engine Instantiation (Corrected Syntax) ---
        // Create 1 Task agent, initialize it, and retrieve it using 'returns'
        create Task number: 1 returns: engines {
            name <- "Biophysical_Engine";
        }
        // Assign the first (and only) agent from the list to the global variable
        biophysical_engine <- first(engines);
        
        // --- 2. Load Data ---
        // Load the CSV content into a matrix (8 rows x 3 columns)
        matrix soil_ref <- matrix(soil_ref_file);
        
        // --- 3. Territory Initialization ---
        ask Parcel {
            // Assign a random soil type (static context s_p)
            self.soil_type_id <- rnd(1, S);
            
            // Fetch Gaussian parameters from the reference matrix
            // Matrix index is (column, row). Row is (soil_type_id - 1)
            int ref_row <- self.soil_type_id - 1;
            float mu_init <- float(soil_ref[1, ref_row]);    // Column 1: mu_soc
            float sigma_init <- float(soil_ref[2, ref_row]); // Column 2: sigma_soc
            
            // Sample initial Soil Organic Carbon (Mg/ha)
            self.SOC <- max(0.0, gauss(mu_init, sigma_init));
            
            // --- VISUALIZATION INIT ---
            // Green Gradient: Hue=120° (approx 0.33), Saturation depends on SOC (capped at 60 Mg/ha for display)
            self.color <- hsb(0.33, min(1.0, self.SOC / 30.0), 0.9);
            
            // Initialize dynamic variables
            self.new_tree_age <- 0;
            self.yield <- 0.0;
            self.delta_SOC <- 0.0;
        }

        // --- 4. Actor Setup ---
        create Actor {
            id <- "Farmer_Agent_1";
            ownedResources <- list(Parcel);
        }
        
        // --- 5. Sync Parameters ---
        // Sync global lists with GUI parameters for the Behavior logic
        soc_effects <- [mu_delta_soc_fallow, mu_delta_soc_manure, mu_delta_soc_tree, mu_delta_soc_baseline];
        yield_effects <- [mu_yield_fallow, mu_yield_manure, mu_yield_tree, mu_yield_baseline];
        
        list<int> obs_init <- Parcel collect ((each.soil_type_id - 1) * (M + 1) + each.new_tree_age);
    	list<float> rewards_init <- Parcel collect (0.0); // Pas de reward au temps 0
    	list<float> socs_init <- Parcel collect each.SOC;
    	list<float> yields_init <- Parcel collect each.yield;
    	
    	ask gym_interface {
        do update_interface(obs_init, rewards_init, socs_init, yields_init);
    }
    
    write "GAMA: Initial state synced. Mean SOC: " + mean(socs_init);
    }
    
    // -------------------------------------------------------------------------
    // --- BATCH SERVER LOGIC (API for Python) ---
    // -------------------------------------------------------------------------
    
	action set_actions(list<int> new_actions) {
        gym_interface.next_action <- new_actions;
    }

    reflex simulation_cycle {
        // On récupère les actions envoyées par Python
        list<int> actions_list <- gym_interface.next_action;
        
        // Sécurité : on ne fait rien si la liste est vide (cas du premier cycle)
        if (length(actions_list) = length(Parcel)) {
            
            ask Parcel {
                int my_action <- actions_list[self.index]; 
                ask biophysical_engine {
                    do execute_on(myself, my_action, soc_effects, yield_effects, sigma_soc, sigma_yield);
                }
                self.color <- hsb(0.33, min(1.0, self.SOC / 30.0), 0.9);
            }
    	    
    	    // Mise à jour des indicateurs
    	    list<int> obs <- Parcel collect ((each.soil_type_id - 1) * (M + 1) + each.new_tree_age);
    	    list<float> rewards <- Parcel collect (alpha * each.delta_SOC + (1.0 - alpha) * each.yield);
    	   
    	    ask gym_interface {
    	        do update_interface(obs, rewards, Parcel collect each.SOC, Parcel collect each.yield);
    	    }
        }
    }
}

species GymAgent {
    map<string, unknown> action_space;
    map<string, unknown> observation_space;
    
    unknown state;
    float reward;
    bool terminated <- false;
    bool truncated <- false;
    map<string, unknown> info;
    
    list<int> next_action; // Python écrira ici
    map<string, unknown> data; // Python lira ici

    // Cette action sera appelée par GAMA après chaque pas de simulation
    action update_interface(list<int> obs, list<float> rewards, list<float> socs, list<float> yields) {
        state <- list(obs);
        reward <- sum(rewards);
        info <- ["soc_map"::socs, "yield_map"::yields];
        data <- ["State"::state, "Reward"::reward, "Terminated"::terminated, "Truncated"::truncated, "Info"::info];
    }
}

experiment "AgroCarbonSimulation" type: gui {
	
	// --- Policy Parameters ---
    parameter "Alpha (Reward Trade-off): " var: alpha min: 0.0 max: 1.0 category: "Policy";
    
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