# Inventory-Optimization
# **Inventory Lot Size Optimization tool** devloppped in the scope of my master's thesis

This tool was developped in order to address the need of a company to have a standardized process of performing inventory lot size optimization. A mathematical model was created and later implemented with the help of the [PuLP](https://coin-or.github.io/pulp/) library. The model can be solved using various open-source and commercial [solvers](https://coin-or.github.io/pulp/technical/solvers.html#module-pulp.apis). Additional capabilitites include the ability to resolve a model stored in memory while modifying some of the constraints and a bi-objective extension of the mathematical model which is then solved using the [epsilon constraint](https://engineering.purdue.edu/~sudhoff/ee630/Lecture09.pdf).



## Mathematical Model
### Introduction
This model was intended to solve the problem of Company X which suffered from high level of inventories while providing low service levels. For this purpose, a multi-commodity, capacitated, three-echelons inventory lot sizing model will be presented. This model considers holding costs, production costs, transportation costs between the facilities and shortage costs occurring from lost sales while respecting the holding capacity limitations at the different facilities and production capacity at the first echelon. 

Throughout the model, the unit for expressing the amount of product will be in pallets which do not have to be entire pallets (i.e., no integer requirement). The products have to be produced at the first echelon whilst taking lead time into consideration if the product in question is internally produced. This lead time will vary depending on the product. Enough quantities should be produced in order to satisfy demand occurring at the producing facilities and the shipments required by the lower echelon in order to satisfy their respective demand. 

At the second echelon, shipments sent in the prior time periods are received which should be enough to cover for the demand occurring at that location, the shipments needed at the last echelon and the safety stock requirements which can be offset by the inventory carried over from the previous time period.

Finally, warehouses belonging to the last echelon receive shipments from the preceding echelons in quantities which should again cover their demand together with a smaller safety stock requirement. There is also the possibility to transfer the inventory from previous time periods. 

The origin and destination of the shipments is determined by the cost matrix of sending a full-truck load between those two locations given that the capacity constraints have not been reached. Additionally, the shipments occur in multiple of 33 which represent the maximum number of pallets transported in a full-truck load.

Further, a distinction is made between products produced by Company X and those produced by external suppliers. For internal SKUs, the lead time is accounted for the first echelon whilst for external SKUs it is assumed that a lead time of one time period (one week) is sufficient.

### Sets

    The set of time period is given by:
t ∊ T = {1, 2,…,|T|}

	The set of products is indexed by:
i ∊ I = {1, 2,…,|I|}

	The subset of products produced internally is indexed by:
k ∊ K = {1, 2,…,|K|}

	The subset of products supplied externally is indexed by:
e ∊ E = {1, 2,…,|E|}

	The set of facilities is indexed by:
w ∊ W = {1, 2,..,|W|}

    The subset of facilities in the first echelon is indexed by:
f ∊ F = {1, 2,…,|F|} and F ⊆ W

	the subset of central warehouses is indexed by:
d ∊ D = {1, 2,…,|D|} and D ⊆ W

    The subset of regional warehouses is indexed by:
j ∊ J = {1, 2,… ,|J|} and J ⊆ W
	
    The set of all external factories is indexed by:
o ∊ O = {1, 2,…,|O|}

    The set of factories that are allowed to receive shipments from other factories is denoted by:
g ∊ G = {1, 2, …,, |G|} and G ⊆ F


### Variables
In order to account for the inventory present in the system, a variable is introduced which reflects the inventory levels present at the end of a time period. For all, i ∊ I, w ∊ W and t ∊ T, let

i<sub>iwt</sub>: Inventory level for product I at facility w at the end of period t

The decision variable representing the shipments between the facilities is defined for all i ∊ I, o ∊ W, d ∊ W and t ∊ T, let

s<sub>odit</sub>: Shipment for product i from facility o to facility d at time t

Since costs are computed on a full-truck load basis, the shipments need to be aggregated per FTL, this variable represents the number of full-truck load going between two facilities during a certain time period which is given by, for all o ∊ W, d ∊ W and t ∊ T let

f<sub>odt</sub>: the amount of full truck load going between facility o and d at time t

The production is also accounted for in order to measure production costs as well as to ensure production capacity is not exceeded. For all, k ∊ K, f ∊ F and t ∊ T let
p<sub>fkt</sub>: Production of product i at facility f at time t

As the minimum amount of a production run is restricted, a binary variable indicating whether production is taking place or not is needed. For all k ∊ K, t ∊ T let

ykt: 1 if the production of product k is taking place in time period t, 0 otherwise
	
  The last variable will measure by how much (in pallets) the demand was not satisfied for a given period. For all, I ∊ I, w ∊ W and t ∊ T, let 

ls<sub>iwt</sub>: lost sales for product I at facility w at time t 


### Parameters
Parameters
	Holding costs are dependent per product and location at which they are being held. Within this model, they are denoted by:

Ch<sub>iw</sub>: holding costs for product i at facility w

Production costs only vary per product since each product has a unique production site. They are denoted by:

Cp<sub>i</sub>: production costs of product i

Transportation costs are defined on a full-truck load basis. They are denoted by:

D<sub>od</sub>: cost of sending a full-truck load from facility o to facility d

The shortage costs need to be defined in order to penalise not meeting the demand. As this value is purely theoretical it is equal for all products. It is given by:

C<sub>s</sub>: shortage cost when demand is not met

The lead time for each product was provided and varies depending on the product. In the model, lead time is only considered for internally produced articles as it assumed that other suppliers always have enough products on hand. The following notation is used:
lt<sub>i</sub>: lead time for product k

In this model, safety stock is a parameter.  It is given by:

SS<sub>iw</sub>: safety stock level for product i at facility w

SS = z\sqrt{LT\sigma_d^2+d^2\sigma_{LT}^2}



