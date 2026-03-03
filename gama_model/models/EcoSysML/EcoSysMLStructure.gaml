/**
* Name: EcoSysMLStructure
* Defines the static elements of a socio-ecological system. 
* Author: Cheikhou Akhmed KANE
* Tags: 
*/

model EcoSysMLStructure

/* Insert your model definition here */

/* 
 * The Actor refers to human agents that perform activities. It is
linked with Resource. It is assigned goals via the concept of Goal.
 */ 
species Actor {
	string id;
	list<Resource> ownedResources;
}

/*
 * The Resource concept represents the resources that actors use. It is
associated with the concept Actor through the ownedResources association. The
concepts AnimalResource, CognitiveResource, MaterialResource and NaturalRe-
source inherit from it.
 */
species Resource {
    string name;
}

/*The NaturalResource concept represents natural elements
that actors use. The concepts Water, Tree, and Land inherit from this concept. 
* */
species NaturalResource parent: Resource {
    string type;
    float quantity;
}

/*
 * The Land concept represents areas of soil used by actors for grazing, cul-
tivation, or settlement. In GAMA, we define it as a grid where each cell is a spatial unit.
 */
species Land parent: NaturalResource {
	string type -> "Land";
}

species Tree parent: NaturalResource {
	string type -> "Tree";
	
	float size <- 1.0 ;
    rgb color <- #green;
	
	aspect base {
	draw circle(size) color: color ;
    }
}

/*
 * The Parcel concept represents a spatial unit of land that can be used
for agricultural, pastoral, or other land-based activities.
 */
grid Parcel parent: Land width: 10 height: 10 {
	
	// State variables from gym-agro-carbon
    int soil_type_id;      // Static context s(p)
    int new_tree_age;       // Dynamic context tau(p) for agent-planted trees
    float SOC;          // Soil Organic Carbon (Natural Resource quantity)
    
    // Output variables (AgriWorkProducts)
    float yield;        // Socio-economic benefit (Yield)
    float delta_SOC;    // Environmental benefit (Carbon Sequestration)
}
