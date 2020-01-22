# MLEC_invoicing

Invoicing in local energy communities (LEC).

# Usage

invoicing.py defines a class Invoicer. Instantiate an object of this class and call the compute_invoice method.

See invoicing.py's __main__ section for testing.

Requires the ipopt solver.

# Conventions

 - Inputs and outputs are either in kW (energy flows) or in EUR/kWh (prices).
 - The intended usage is for market periods of 15 minutes (the case of Belgium) (it should normally work with other values by changing the value of the parameter MARKET_PERIOD_DURATION_IN_MINUTES.
 - Timestamps always indicate the start time of a period.
 - The resolution of the input data must divide the market period in an integer number of periods (e.g. 30 sec or 1 min) and be synchronized with market periods.
 - If an entity (or member of the LEC) has a storage device, it is assumed that the storage device captures all the flows injected or withdrawn by this entity (this is for simplification, and this is the case in the Merygrid project).
 - Over a time-step at the resolution of the input data, it is considered that an entity is either injecting or withdrawing (not both). This is obviously not the case in the output at the market period resolution.


# Input format
The Invoicer class's constructor has 3 arguments:
 1. **structure** describes the LEC configuration and parameters. It is a dictionary with keys
   - community_tariff: tariff for exchanges in the community in EUR/kWh, payed both for imports and imports. E.g. if 1 kWh is send from a member to another over a market period, 2 * community_tariff is perceived by the community operator.
   - community_past_peak: peak recorded for the community (i.e. at the substation), in kW, since the beginning of the month.
   - community_peak_price: peak price to apply to the community in EUR/kW. Note: if not positive, peak values reported will have no meaning => use a tiny value even if you do not want
     to penalize the peak.
   - entities: a list of the entities constituting the community. Each entity is described by
     - entity_id: a integer, unique identifier of the entity.
     - EAN_inj: a string representing the injection EAN of the entity (if any).
     - EAN_wd: a string representing the withdrawal EAN of the entity (if any).
     - past_peak: past peak [kW] value recorded during the same month for the entity *alone*.
     - peak_price: in EUR/kW, price for peak penalty for entity alone. Note: if not positive, peak values reported will have no meaning => use a tiny value even if you do not want to penalize the peak.
     - storage: a dictionary containing the information to construct the storage device of the entity (if any). A
       storage is described by:
        - usage_fee: storage usage fee, i.e. capex cost per kWh. Too high -> storage not used. Too low -> storage not enough remunerated.
        - charge_efficiency: between 0 and 1. Only kWh effectively stored and "restored" are valued at the usage_fee.
        - discharge_efficiency: between 0 and 1.
  2. **data** is a dataframe indexed by time steps (format is YYYY-mm-ddTHH:MM:SS). It must contain
   - one column per EAN of the structure, in kW.
   - one "sale_price_%d" % entity_id column per entity indicating the price it can sell to the grid [EUR/kWh].
   - one "purchase_price_%d" % entity_id column per entity indicating the price it can buy from the grid [EUR/kWh].
  3. **n_periods_in_quarter** is the number of measurements within a market period. Defaults to 30. Must be coherent with the input data (no check is performed).


# Output format
The output is a dictionary with entity ids (int) as keys and pandas dataframe as values.
The dataframes are indexed by market period (market period start time, as usual). Each dataframe has 5 columns: 
 - price_com: average price at which exchanges are priced for the entity during the quarter (both for imports and exports).
 - exp_grid: power injected in the grid.
 - imp_grid: power withdrawn from the grid.
 - exp_com: power injected in the community.
 - imp_com: power withdrawn from the community.

# Methodology
The approach is very similar to what is proposed in 

> B. Cornélusse, I. Savelli, S. Paoletti, A. Giannitrapani, A. Vicino, A community microgrid architecture with an internal local market, Applied Energy 242 (2019) 547– 560.

with the following adaptations. 
 - A problem is run by market period and the only decisions variables are the power injected in /withdrawn from the community and the grid. 
 - Storage fees are modeled with the limitation explained above. 
 - No reserve functionality is implemented. 
 - The peak model accounts for past peaks, i.e. only accounts a peak penalty if a peak during the day exceeds a peak that occurred the same month.
 - A 2-phase approach is used for the profit repartition. The model solves a first problem where the Pareto superior condition ensures the profit of a member in the community is at least as good as the profit alone. Then a second phase is run to attempt to equalize the profits over entities (max margin approach, see the alpha variable).
 - A run with with 96 market periods of 15 minutes divided in 30 seconds intervals takes around 1 min on a Macbook.