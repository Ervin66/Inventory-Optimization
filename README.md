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

<img src="https://latex.codecogs.com/svg.latex?SS\&space;=\&space;z\sqrt{LT\sigma_d^2&plus;d^2\sigma_{LT}^2}" title="SS\ =\ z\sqrt{LT\sigma_d^2+d^2\sigma_{LT}^2}" />

### Objective

Let the holding costs be defined by:
<img src="https://latex.codecogs.com/svg.latex?C_1\left(i\right)=\sum_{i\&space;\in&space;I}\sum_{w\in&space;W}\sum_{t\in&space;T}{Ch_{iw}i_{iwt}}" title="C_1\left(i\right)=\sum_{i\ \in I}\sum_{w\in W}\sum_{t\in T}{Ch_{iw}i_{iwt}}" />

The transportation costs are then given by the following function:
<img src="https://latex.codecogs.com/svg.latex?C_2\left(f\right)=\sum_{o\&space;\in&space;W}\sum_{d\in&space;W}\sum_{t\in&space;T}&space;f_{odt}D_{od}" title="C_2\left(f\right)=\sum_{o\ \in W}\sum_{d\in W}\sum_{t\in T} f_{odt}D_{od}" />

The production costs are then given by:
<img src="https://latex.codecogs.com/svg.latex?C_3\left(p\right)=\sum_{i\&space;\in&space;I}\sum_{f\in&space;F}\sum_{t\in&space;T}{C{pk}_fp_{fkt}}" title="C_3\left(p\right)=\sum_{i\ \in I}\sum_{f\in F}\sum_{t\in T}{C{pk}_fp_{fkt}}" />

The following cost function representing the stock-out costs is devised:
<img src="https://latex.codecogs.com/svg.latex?C_4\left(ls\right)=\sum_{i\in&space;I}\sum_{w\in&space;W}{\sum_{t\in&space;T}&space;C&space;s&space;l&space;s_{iwt}}" title="C_4\left(ls\right)=\sum_{i\in I}\sum_{w\in W}{\sum_{t\in T} C s l s_{iwt}}" />

The different cost functions are then combined to form the objective function of a Mixed-Integer Linear Program (MILP)
<img src="https://latex.codecogs.com/svg.latex?C\left(i,f,p,ls\right)=C_1\left(i\right)&plus;C_2\left(f\right)&plus;C_3\left(p\right)&plus;C_4\left(ls\right)" title="C\left(i,f,p,ls\right)=C_1\left(i\right)+C_2\left(f\right)+C_3\left(p\right)+C_4\left(ls\right)" />

### Constraints

In order to ensure that demand is satisfied at the first echelon as well as enough that enough internal SKUs products will be shipped to the following echelons, the following constraint is needed:
<img src="https://latex.codecogs.com/svg.latex?i_{kft-1}\&space;&plus;p_{fkt-lt_i}-d_{kft}-\sum_{d\in&space;D}&space;s_{fdkt}\&space;-\sum_{j\in&space;J}&space;s_{fjkt}\&space;\geq&space;i_{kft}-{ls}_{kft}\&space;\&space;\&space;\&space;\&space;\&space;\forall&space;t>lt_k,f\in&space;F,k\in&space;K" title="i_{kft-1}\ +p_{fkt-lt_i}-d_{kft}-\sum_{d\in D} s_{fdkt}\ -\sum_{j\in J} s_{fjkt}\ \geq i_{kft}-{ls}_{kft}\ \ \ \ \ \ \forall t>lt_k,f\in F,k\in K" />

In order to produce meaningful results in the initial time periods, lead time is assumed to be of one week until the time period being iterated over surpasses the lead time for that SKU. In practice, this gives place to the following constraint:
<img src="https://latex.codecogs.com/svg.latex?i_{kft-1}\&space;&plus;p_{fkt-1}-d_{kft}-\sum_{d\in&space;D}&space;s_{fdkt}\&space;-\sum_{j\in&space;J}&space;s_{fjkt}\&space;\geq&space;i_{kft}-{ls}_{kft}&space;\forall\&space;t>0&space;\cup&space;t\le&space;lt_k,&space;f\in&space;F,k\in&space;K" title="i_{kft-1}\ +p_{fkt-1}-d_{kft}-\sum_{d\in D} s_{fdkt}\ -\sum_{j\in J} s_{fjkt}\ \geq i_{kft}-{ls}_{kft} \forall\ t>0 \cup t\le lt_k, f\in F,k\in K" />

For the next echelon, a similar constraint is required, which will provide its inventory balance. The shipments from the echelon above sent in the previous time period are received while shipment sent to the next echelon are deducted. This is where most of the safety stock is also held which gives the following constraint:
<img src="https://latex.codecogs.com/svg.latex?i_{kdt-1}-\sum_{j\in\&space;J}&space;s_{djkt-1}\&space;&plus;\sum_{f\in\&space;F}&space;s_{fdkt-1}\&space;-d_{kdt}\geq\&space;i_{kdt}\&space;\&space;-\&space;{ls}_{kdt}\&space;\&space;\&space;\&space;\&space;\forall\&space;t>0,d\in\&space;D,k\in\&space;K" title="i_{kdt-1}-\sum_{j\in\ J} s_{djkt-1}\ +\sum_{f\in\ F} s_{fdkt-1}\ -d_{kdt}\geq\ i_{kdt}\ \ -\ {ls}_{kdt}\ \ \ \ \ \forall\ t>0,d\in\ D,k\in\ K" />

The third echelon inventory balance follows a similar logic. The major difference is that shipments can only be received as this is the last echelon. As prior, enough products should be supplied to satisfy the SKU specific demand, this is given by the following constraint:
<img src="https://latex.codecogs.com/svg.latex?i_{kdt-1}&plus;\sum_{j\in\&space;J}&space;s_{djkt-1}\&space;&plus;\sum_{f\in\&space;F}&space;s_{fdkt-1}\&space;-d_{kdt}\geq\&space;i_{kdt}\&space;\&space;-\&space;{ls}_{kdt}\&space;\&space;\&space;\&space;\&space;\forall\&space;t>0,j\in\&space;J,k\in\&space;K" title="i_{kdt-1}+\sum_{j\in\ J} s_{djkt-1}\ +\sum_{f\in\ F} s_{fdkt-1}\ -d_{kdt}\geq\ i_{kdt}\ \ -\ {ls}_{kdt}\ \ \ \ \ \forall\ t>0,j\in\ J,k\in\ K" />

. In certain cases, demand takes place for articles in factories that are not produced at said location. This usually occurs when orders are placed for multiple articles at a facility of the first echelon. By allowing it, shipments can be consolidated. This is the only case where lateral shipments can happen. The inventory balance for lateral shipments is given by:
<img src="https://latex.codecogs.com/svg.latex?i_{kgt-1}&plus;\sum_{f\in&space;F}&space;s_{fgkt-1}-d_{kgt}\geq&space;i_{kgt}-ls_{kgt\&space;\&space;\&space;}\forall&space;t>lt_k,g\in&space;G,k\in&space;K" title="i_{kgt-1}+\sum_{f\in F} s_{fgkt-1}-d_{kgt}\geq i_{kgt}-ls_{kgt\ \ \ }\forall t>lt_k,g\in G,k\in K" />

Next, the inventory balance constraints that will treat external SKUs need to be defined. The principle remains largely the same but the variable involved slightly differs as the supply is no longer insured by the first echelon but by shipment originating from outside of Company X’s network. For the first echelon, the inventory balance is given by the following constraint:
<img src="https://latex.codecogs.com/svg.latex?i_{eft-1}\&space;-d_{eft}&plus;\sum_{o\in\&space;O}&space;s_{ofet}\&space;\&space;\geq\&space;i_{eft}-{ls}_{eft}\&space;\&space;\&space;\&space;\&space;\&space;\&space;\forall\&space;t>0,f\in\&space;F,e\in\&space;E" title="i_{eft-1}\ -d_{eft}+\sum_{o\in\ O} s_{ofet}\ \ \geq\ i_{eft}-{ls}_{eft}\ \ \ \ \ \ \ \forall\ t>0,f\in\ F,e\in\ E" />

At the second echelon, the equation is similar as in Equation 35, where shipments from the first echelon are replaced by external shipments, it can be formulated as such:
<img src="https://latex.codecogs.com/svg.latex?i_{edt-1}-\sum_{j\in\&space;J}&space;s_{djet-1}\&space;&plus;\sum_{o\in\&space;O}&space;s_{odet-1}\&space;-d_{edt}\geq\&space;i_{edt}&space;-&space;{ls}_{edt}\&space;\&space;\&space;\forall\&space;t>0,d\in\&space;D,e\in&space;E" title="i_{edt-1}-\sum_{j\in\ J} s_{djet-1}\ +\sum_{o\in\ O} s_{odet-1}\ -d_{edt}\geq\ i_{edt} - {ls}_{edt}\ \ \ \forall\ t>0,d\in\ D,e\in E" />

The inventory balance for external products at the third echelon is given by:
<img src="https://latex.codecogs.com/svg.latex?i_{ejt-1}&plus;\sum_{d\in\&space;D}&space;s_{djet-1}\&space;&plus;\sum_{o\in\&space;O}&space;s_{ojet-1}\&space;-d_{ejt}\geq\&space;i_{ejt}\&space;\&space;-\&space;{ls}_{ejt}\&space;\&space;\&space;\&space;\&space;\forall\&space;t>0,j\in\&space;J,e\in\&space;E" title="i_{ejt-1}+\sum_{d\in\ D} s_{djet-1}\ +\sum_{o\in\ O} s_{ojet-1}\ -d_{ejt}\geq\ i_{ejt}\ \ -\ {ls}_{ejt} \forall\ t>0,j\in\ J,e\in\ E" />

In order to ensure that the safety stock is held at the relevant locations, the next constraint is introduced: 
<img src="https://latex.codecogs.com/svg.latex?i_{iwt}\geq&space;SS_{iw}-{ls}_{iwt}\forall&space;t>0,w\in&space;D\cup&space;J,i\in&space;I" title="i_{iwt}\geq SS_{iw}-{ls}_{iwt}\forall t>0,w\in D\cup J,i\in I" />

The initial inventory is set to be enough for the demand occurring in the first period and the eventual safety stock requirement, this is given by the following equation:
<img src="https://latex.codecogs.com/svg.latex?i_{iw0}=d_{iw0}&plus;SS_{iw}&space;\&space;\&space;\forall&space;t=0,\&space;w\in&space;W,\&space;i\in&space;I" title="i_{iw0}=d_{iw0}+SS_{iw} \ \ \forall t=0,\ w\in W,\ i\in I" />

In order to define the variable representing the full-truck load all shipments at given time between a given origin-destination pair are aggregated and divided by 33, the following formula is used:
<img src="https://latex.codecogs.com/svg.latex?\sum_{i\in&space;I}\frac{1}{33}s_{odit}=f_{odt}\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\forall&space;o\in&space;W,d\in&space;W,t\in&space;T" title="\sum_{i\in I}\frac{1}{33}s_{odit}=f_{odt}\ \ \ \ \ \ \ \ \ \ \ \ \forall o\in W,d\in W,t\in T" />

As the amount produced cannot exceed the capacity at each of the facilities member of the first echelon, the next constraint is needed:
<img src="https://latex.codecogs.com/svg.latex?\sum_{k\in&space;K}&space;p_{fkt}\le&space;Pc_f\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\forall&space;f\in&space;F,t\in&space;T" title="\sum_{k\in K} p_{fkt}\le Pc_f\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \forall f\in F,t\in T" />

Similarly, the holding capacity for all facilities in the supply chain needs to be respected. This is defined by the following constraint:
<img src="https://latex.codecogs.com/svg.latex?\sum_{i\in&space;I}&space;i_{iwt}\le&space;Hc_w\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\forall&space;w\in&space;W,t\in&space;T" title="\sum_{i\in I} i_{iwt}\le Hc_w\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \forall w\in W,t\in T" />

The following two constraints ensure that the minimum batch production size is respected and a big-M constraint is also needed for the first constraint to hold:
<img src="https://latex.codecogs.com/svg.latex?p_{kft}\geq&space;y_{kt}MB_{k\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;}\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\forall&space;f\in&space;F,k\in&space;K,t\in&space;T" title="p_{kft}\geq y_{kt}MB_{k\ \ \ \ \ \ \ \ \ }\ \ \ \ \ \ \ \ \ \ \ \ \ \ \forall f\in F,k\in K,t\in T" />

<img src="https://latex.codecogs.com/svg.latex?p_{kft}\le&space;y_{kt}M\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\forall&space;f\in&space;F,k\in&space;K,t\in&space;T" title="p_{kft}\le y_{kt}M\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \forall f\in F,k\in K,t\in T" />

The last set of constraints ensure that non-negativity and binarity in the case of the production setup variable.
<img src="https://latex.codecogs.com/svg.latex?i_{iwt},{ls}_{iwt},s_{odit},f_{odt},p_{fit}\geq0\&space;\&space;\&space;\&space;\&space;\&space;\&space;\forall&space;i\in&space;I,\left(w,o,d,f,j\right)\in&space;W,t\in&space;T" title="i_{iwt},{ls}_{iwt},s_{odit},f_{odt},p_{fit}\geq0\ \ \ \ \ \ \ \forall i\in I,\left(w,o,d,f,j\right)\in W,t\in T" />

<img src="https://latex.codecogs.com/svg.latex?y_{kt}\in0,1\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\forall&space;k\in&space;K,t\in&space;T" title="y_{kt}\in0,1\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \forall k\in K,t\in T" />

### bi-objective extension

The already formulated model while containing different cost function components, deals solely with improving economical performances but it is worthwhile to enlarge the focus of the model to other business considerations. For this purpose, a second objective will be added which this time considers the service level provided by a certain solution. 
	For this purpose, a new variable is introduced which will measure the service level:
sl: the average service level over the complete planning horizon

Complimentary to this new variable, a new constraint is required to reflect its purpose:
<img src="https://latex.codecogs.com/svg.latex?\sum_{i\in&space;I}\sum_{w\in&space;W}\sum_{t\in&space;T}{ls}_{iwt}\&space;=\&space;\sum_{i\in&space;I}\sum_{w\in&space;W}\sum_{t\in&space;T}{d_{iwt\&space;}(1-sl)}" title="\sum_{i\in I}\sum_{w\in W}\sum_{t\in T}{ls}_{iwt}\ =\ \sum_{i\in I}\sum_{w\in W}\sum_{t\in T}{d_{iwt\ }(1-sl)}" />

The epsilon constraint is then implemented this way:
<img src="https://latex.codecogs.com/svg.latex?sl&space;>&space;\epsilon_{i}" title="sl > \epsilon_{i}" />

until the model is no longer solvable.





