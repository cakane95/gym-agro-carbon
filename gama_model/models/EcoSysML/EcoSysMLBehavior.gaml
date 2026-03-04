/**
* Name: EcoSysMLBehavior
* Defines the dynamic aspects of a socio-ecological system. 
* It provides the elements needed to model the behavior of actors within an SES, describing how they interact with their environment and make decisions based on various processes, constraints, and learning mechanisms.
* Author: Cheikhou Akhmed KANE
* Tags: 
*/


model EcoSysMLBehavior

import "EcoSysMLStructure.gaml"

/* Insert your model definition here */

/*
 * The Activity concept represents actions undertaken by an actor within
the socio-ecological system. Each Activity is performed by an Actor and consists
of one or more Tasks.
 */
species Activity {
	string name;
    list<Task> ownedTasks; // Composition of tasks
}

/*
 * The Task concept represents the smallest unit of work within an Activity.
Tasks are executed as part of a larger activity.
 */
species Task {
	string name;
	
	action execute_on(Parcel cell, int action_id, list<float> soc_vals, list<float> yield_vals, float s_soc, float s_yield) {
        
        // 1. Retrieve expected values (mu)
        float mu_soc <- soc_vals[action_id];
        float mu_yield <- yield_vals[action_id];
        
        // 2. Apply transition with Gaussian noise (Stochasticity)
        // SOC update: Mg/ha
        float noise_soc <- gauss(0, s_soc);
        cell.SOC <- cell.SOC + mu_soc + noise_soc; 
        
        // Yield update: t/ha (Ensure yield is not negative)
        float noise_yield <- gauss(0, s_yield);
        cell.yield <- max(0.0, mu_yield + noise_yield);
        
        // 3. Resource Tree Management (tau update)
        if (action_id = 2) { 
            if (cell.new_tree_age = 0) {
                cell.new_tree_age <- 1;
                create Tree { location <- cell.location; }
            }
        }
        
        // Seasonal aging of agent-planted trees
        if (cell.new_tree_age > 0 and cell.new_tree_age < 20) {
            cell.new_tree_age <- cell.new_tree_age + 1;
        }
        
        // 4. Output AgriWorkProduct (Carbon Sequestration)
        cell.delta_SOC <- mu_soc + noise_soc; 
    }
}

species LearningModel {
	string type; // "Reinforcement Learning" 
    string algorithm; // UCB, TS, NPTS
}