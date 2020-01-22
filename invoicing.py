from datetime import timedelta

import pandas as pd
from typing import Dict

from pyomo.core import ConcreteModel, Var, NonNegativeReals, Set, RangeSet, Reals, Objective, Constraint, maximize
import pyomo.environ  # REQUIRED, DO NOT REMOVE
from pyomo.opt import TerminationCondition, SolverStatus, ProblemFormat

DEFAULT_PEAK_PRICE = 0.001  # In EUR/kW used only if none is provided in input json.
MARKET_PERIOD_DURATION_IN_MINUTES = 15  # No comment.
MARKET_PERIOD_DURATION_IN_HOURS = MARKET_PERIOD_DURATION_IN_MINUTES / 60.  # No comment.
# NB: the var name quarter means "market_period in the code, for short.
DEFAULT_IPOPT_TOL = 1e-8  # Default global tolerance parameter used with ipopt


def is_zero(val: float):
    return abs(val) < 1e-5


class SolverError(Exception):
    pass


class Storage:
    def __init__(self, usage_fee: float, charge_efficiency: float, discharge_efficiency: float):
        """
        Data access class.
        :param usage_fee: Storage usage fee, i.e. capex cost per kWh.
        :param charge_efficiency: between 0 and 1.
        :param discharge_efficiency: between 0 and 1.
        """
        self.usage_fee = usage_fee
        assert (self.usage_fee >= 0.)
        self.charge_efficiency = charge_efficiency
        assert (0. <= self.charge_efficiency <= 1.)
        self.discharge_efficiency = discharge_efficiency
        assert (0. <= self.discharge_efficiency <= 1.)


class Entity:
    def __init__(self, entity_id: int, EAN_inj: str = None, EAN_wd: str = None, storage: dict = None,
                 past_peak: float = 0., peak_price: float = DEFAULT_PEAK_PRICE):
        """
        Data access class
        :param entity_id: unique identifier of the entity.
        :param EAN_inj: string representing the injection EAN of the entity (if any).
        :param EAN_wd: string representing the withdrawal EAN of the entity (if any).
        :param storage: dictionary containing the information to construct the storage device of the entity (if any).
        :param past_peak: past peak [kW] value recorded during the same month for the entity *alone*, updated by the algorithm to a higher value if a new peak is reached during the day.
        :param peak_price: in EUR/kW.
        """
        self.entity_id = entity_id
        self.EAN_inj = EAN_inj
        self.EAN_wd = EAN_wd
        self.storage = Storage(**storage) if (storage is not None) else None
        if self.storage is not None:
            assert ((self.EAN_inj is not None) and (self.EAN_wd is not None))
        self.past_peak = past_peak
        assert (self.past_peak >= 0.)
        self.peak_price = peak_price
        assert (self.peak_price >= 0.)


class Invoicer:
    def __init__(self, structure: dict, data: pd.DataFrame, n_periods_in_quarter: int = 30):
        """
        :param structure: list of entities, and community_tariff
        :param data: measured total production [kW] and consumption [kW] of entities and grid purchase and sell prices at a resolution
        of n_periods_in_quarter for one day (from 00:00:00 to 23:59:00)
        :param n_periods_in_quarter:
        """

        self.structure = structure
        self.data = data
        self.n_periods_in_quarter = n_periods_in_quarter
        self.period_duration_in_minutes = 15 / self.n_periods_in_quarter
        self.period_duration_in_hours = self.period_duration_in_minutes / 60.  # For conversion of kW in kWh
        self.community_tariff = self.structure["community_tariff"]
        assert (self.community_tariff >= 0.)
        self.community_past_peak = self.structure["community_past_peak"]
        assert (self.community_past_peak >= 0.)
        self.community_peak_price = self.structure[
            "community_peak_price"] if "community_peak_price" in self.structure else DEFAULT_PEAK_PRICE
        assert (self.community_peak_price >= 0.)

        # Instantiate the input and ouput variables
        self.entities = {e["entity_id"]: Entity(**e) for e in self.structure["entities"]}
        self.invoices: dict = {e: [] for e in self.entities}
        from pyomo.opt import SolverFactory
        self.opt = SolverFactory("ipopt")
        self.ipopt_tol = DEFAULT_IPOPT_TOL
        self.opt.options.update({"tol": self.ipopt_tol})
        self.comparison_df = {e: [] for e in self.entities}

    def compute_invoice(self) -> Dict[int, pd.DataFrame]:
        """
        Computes the repartition between community and grid flows and the community prices to create the invoice.
        As byproduct, stores profit_alone and profit_community per entity in self.comparison_df.
        :return: A dict of entity_id -> dataframes, containing average imports and exports to the grid and to the community
         per quarter, and average community exchange prices per quarter.
        """

        # Loop over quarters
        start_quarter = data.index[0]
        end_quarter = start_quarter + timedelta(minutes=(24 * 60 - 15))
        quarters = pd.date_range(start_quarter, end_quarter, freq="900S")
        assert (len(quarters) == 96)

        for quarter in quarters:
            print(quarter)
            entity_alone_profits = self._compute_entity_alone_profits(quarter)
            results = self._compute_community_repartition(quarter, entity_alone_profits)

            for e in self.entities:
                self.invoices[e].append([quarter] + results[e].values.tolist())

        # convert the results list in pd.Dataframes indexed by datetime
        for e in self.entities:
            self.invoices[e] = pd.DataFrame(data=self.invoices[e],
                                            columns=["DateTime", "price_com", "exp_grid", "imp_grid", "exp_com",
                                                     "imp_com"])
            self.invoices[e].set_index("DateTime", inplace=True)

        return self.invoices

    def _compute_entity_alone_profits(self, quarter: pd.Timestamp) -> Dict[int, float]:
        """
        Profit (or cost) of export, import, peak under balance constraint. No exchange between entities.
        Only one problem gathering all the entities is solved, but no variable is shared among entities.
        :param quarter: current quarter start time.
        :return: A dict entity_id -> entity_alone_profit.
        """

        model = ConcreteModel()

        # Sets
        self._add_sets(model)

        # Variables
        self._add_grid_vars(model)

        model.peak = Var(model.entities, within=NonNegativeReals)  # Entity peak [kW]
        model.J = Var(model.entities, within=Reals)  # Entity profit [EUR]

        def entity_balance_rule(m, e, p):
            # Fix variables depending on the net position. (Note: If position is 0, fix all to zero.)
            if is_zero(self._net_position(e, p, quarter)):
                m.imp_grid[e, p].fix(0)
                m.exp_grid[e, p].fix(0)
                return Constraint.Skip
            elif self._net_position(e, p, quarter) > 0:
                # export position, can fix imports to zero
                m.imp_grid[e, p].fix(0)
            elif self._net_position(e, p, quarter) < 0:
                # import position, can fix exports to zero
                m.exp_grid[e, p].fix(0)

            return m.exp_grid[e, p] - m.imp_grid[e, p] == self._net_position(e, p, quarter)

        model.entity_balance = Constraint(model.entities, model.periods, rule=entity_balance_rule)

        def entity_peak_rule(m, e):
            """

            """
            lhs = sum(m.imp_grid[e, p] for p in model.periods) / self.n_periods_in_quarter
            rhs = self.entities[e].past_peak + m.peak[e]
            return lhs <= rhs

        model.entity_peak = Constraint(model.entities, rule=entity_peak_rule)

        def entity_profit_rule(m: ConcreteModel, e: int):
            expr = 0

            # Energy
            for p in model.periods:
                expr += m.exp_grid[e, p] * self._sale_price(e, p, quarter) * self.period_duration_in_hours
                expr -= m.imp_grid[e, p] * self._purchase_price(e, p, quarter) * self.period_duration_in_hours

            # Capacity
            expr -= m.peak[e] * self.entities[e].peak_price

            return model.J[e] == expr

        model.entity_profit = Constraint(model.entities, rule=entity_profit_rule)

        model.obj = Objective(expr=sum(model.J[e] for e in model.entities), sense=maximize)

        self._solve(model, name="entity")

        entity_alone_profits = {e: model.J[e].value for e in model.entities}
        for e in model.entities:
            self.entities[e].past_peak = max(self.entities[e].past_peak, model.peak[e].value)
            # DEBUG print("Entity %d peak: " % e, self.entities[e].past_peak)

        return entity_alone_profits

    def _compute_community_repartition(self, quarter: pd.Timestamp, entity_alone_profits: Dict[int, float]) -> Dict[
        int, pd.DataFrame]:
        """
        Repartition of profit for a quarter.
        The model solves a first problem where the pareto superior condition ensures the profit in the community is at least as good as the profit alone.
        Then a second phase is run to attemp to equalize the profits over entities (max margin approach).
        :param quarter: current quarter start time.
        :param entity_alone_profits: output of _compute_entity_alone_profits.
        :return: A dict entity_id -> dataframe with average results over quarter.
        """
        model = ConcreteModel()

        # Sets
        self._add_sets(model)

        # Variables
        # Primal
        self._add_grid_vars(model)
        model.exp_com = Var(model.entities, model.periods,
                            within=NonNegativeReals)  # Power sent to the community over period [kW]
        model.imp_com = Var(model.entities, model.periods,
                            within=NonNegativeReals)  # Energy imported from the community over period [kW]
        model.peak_MU = Var(model.entities, within=NonNegativeReals)  # Entity peak, community mode [kW]

        # Profit repartition threshold
        model.alpha = Var(within=NonNegativeReals)

        # Dual
        model.price_com = Var(model.entities, model.periods, within=Reals)
        model.mu = Var(model.periods, within=Reals)
        model.phi_peak = Var(within=NonNegativeReals)

        # Profit
        model.J_MU = Var(model.entities, within=Reals)

        # Expressions
        def profit_expression_rule(m, e):
            expr = 0

            # Energy
            for p in model.periods:
                expr += m.exp_grid[e, p] * self._sale_price(e, p, quarter) * self.period_duration_in_hours
                expr -= m.imp_grid[e, p] * self._purchase_price(e, p, quarter) * self.period_duration_in_hours

                expr += m.exp_com[e, p] * m.price_com[e, p] * self.period_duration_in_hours
                expr -= m.imp_com[e, p] * m.price_com[e, p] * self.period_duration_in_hours

                # Storage
                if self.entities[e].storage is not None:
                    s: Storage = self.entities[e].storage
                    expr += m.exp_com[e, p] / s.discharge_efficiency * s.usage_fee * self.period_duration_in_hours
                    expr += m.imp_com[e, p] * s.charge_efficiency * s.usage_fee * self.period_duration_in_hours

            # Capacity
            expr -= m.peak_MU[e] * self.community_peak_price

            return m.J_MU[e] == expr

        model.profit_MU_def = Constraint(model.entities, rule=profit_expression_rule)

        # Constraints
        # Primal

        def entity_balance_rule(m, e, p):
            # Fix variables depending on the net position. (Note: If position is 0, fix all to zero.)
            if is_zero(self._net_position(e, p, quarter)):
                m.imp_grid[e, p].fix(0)
                m.imp_com[e, p].fix(0)
                m.exp_grid[e, p].fix(0)
                m.exp_com[e, p].fix(0)
                return Constraint.Skip
            elif self._net_position(e, p, quarter) > 0:
                # export position, can fix imports to zero
                m.imp_grid[e, p].fix(0)
                m.imp_com[e, p].fix(0)
            elif self._net_position(e, p, quarter) < 0:
                # import position, can fix exports to zero
                m.exp_grid[e, p].fix(0)
                m.exp_com[e, p].fix(0)

            return m.exp_grid[e, p] - m.imp_grid[e, p] + m.exp_com[e, p] - m.imp_com[e, p] == self._net_position(e, p,
                                                                                                                 quarter)

        model.entity_balance = Constraint(model.entities, model.periods, rule=entity_balance_rule)

        def com_balance_rule(m, p):
            return sum(m.exp_com[e, p] - m.imp_com[e, p] for e in model.entities) == 0

        model.com_balance = Constraint(model.periods, rule=com_balance_rule)

        def global_peak_rule(m):
            """

            :param m: Pyomo model
            :param p: time period
            :return: constraint on the peak of the community
            """
            lhs = sum(m.imp_grid[e, p] for (e, p) in model.entities * model.periods) / self.n_periods_in_quarter
            rhs = self.community_past_peak + sum(m.peak_MU[e] for e in model.entities)
            return lhs <= rhs

        model.global_peak = Constraint(rule=global_peak_rule)

        # Dual
        def dual_exp_grid_rule(m, e, p):
            """
            Dual constraint for the export to grid variable

            :param m: Pyomo model
            :param e: entity
            :param p: time period
            :return: inequality constraint
            """
            return m.price_com[e, p] >= self._sale_price(e, p, quarter)

        model.dual_exp_grid = Constraint(model.entities, model.periods, rule=dual_exp_grid_rule)

        def dual_imp_grid_rule(m, e, p):
            """
            Dual constraint for the import from grid variable

            :param m: Pyomo model
            :param e: entity
            :param p: time period
            :return: inequality constraint
            """
            expr = - m.price_com[e, p] + m.phi_peak / self.period_duration_in_hours
            return expr >= - self._purchase_price(e, p, quarter)

        model.dual_imp_grid = Constraint(model.entities, model.periods, rule=dual_imp_grid_rule)

        def dual_peak_rule(m):
            """
            Dual constraint for the entity peak variables

            :param m: Pyomo model
            :param e: entity
            :return: inequality constraint
            """
            return -m.phi_peak >= - self.community_peak_price

        model.dual_peak = Constraint(rule=dual_peak_rule)

        def dual_exp_com_rule(m, e, p):
            if self._net_position(e, p, quarter) > 0:
                # export position
                return m.price_com[e, p] - m.mu[p] == -self.community_tariff
            else:
                return m.price_com[e, p] - m.mu[p] >= -self.community_tariff

        model.dual_exp_com = Constraint(model.entities, model.periods, rule=dual_exp_com_rule)

        def dual_imp_com_rule(m, e, p):
            if self._net_position(e, p, quarter) < 0:
                # import position
                return - m.price_com[e, p] + m.mu[p] == -self.community_tariff
            else:
                return - m.price_com[e, p] + m.mu[p] >= -self.community_tariff

        model.dual_imp_com = Constraint(model.entities, model.periods, rule=dual_imp_com_rule)

        # Strong duality
        def strong_duality_rule(m):
            """
            Strong duality constraint.

            :param m: Pyomo model
            :return: Equality constraint
            """
            dual_obj = 0

            for p in model.periods:
                for e in model.entities:
                    dual_obj += m.price_com[e, p] * self._net_position(e, p, quarter) * self.period_duration_in_hours

            dual_obj += self.community_past_peak * model.phi_peak

            primal_obj = 0

            for e in model.entities:
                for p in model.periods:
                    # Energy
                    primal_obj += m.exp_grid[e, p] * self._sale_price(e, p, quarter) * self.period_duration_in_hours
                    primal_obj -= m.imp_grid[e, p] * self._purchase_price(e, p, quarter) * self.period_duration_in_hours

                    primal_obj += (m.exp_com[e, p] - m.imp_com[
                        e, p]) * self.community_tariff * self.period_duration_in_hours

                # Capacity
                primal_obj -= m.peak_MU[e] * DEFAULT_PEAK_PRICE

                if self.entities[e].storage is not None:
                    s: Storage = self.entities[e].storage
                    primal_obj += m.exp_com[e, p] / s.discharge_efficiency * s.usage_fee * self.period_duration_in_hours
                    primal_obj += m.imp_com[e, p] * s.charge_efficiency * s.usage_fee * self.period_duration_in_hours

            return dual_obj == primal_obj

        model.strong_duality = Constraint(rule=strong_duality_rule)

        # Upper level

        def pareto_superior_rule(m, e):
            # TODO return (m.J_MU[e] - entity_alone_profits[e]) / abs(entity_alone_profits[e]) >= m.alpha
            return m.J_MU[e] >= entity_alone_profits[e]

        def pareto_superior_rule_phase_2(m, e):
            if not is_zero(entity_alone_profits[e]):
                return (m.J_MU[e] - entity_alone_profits[e]) / abs(entity_alone_profits[e]) >= m.alpha
            else:
                return m.J_MU[e] >= entity_alone_profits[e]

        model.pareto_superior_cdt = Constraint(model.entities, rule=pareto_superior_rule)

        model.objective = Objective(
            expr=sum((model.J_MU[e]) for e in model.entities),
            sense=maximize)

        self._solve(model)

        # PHASE 2 -> max margin profit repartition
        def community_minimum_profit_rule(m):
            return sum((model.J_MU[e]) for e in model.entities) >= model.objective.expr

        model.community_minimum_profit = Constraint(rule=community_minimum_profit_rule)

        model.pareto_superior_cdt.deactivate()
        model.pareto_superior_cdt_phase2 = Constraint(model.entities, rule=pareto_superior_rule_phase_2)

        model.objective.deactivate()
        model.objective_phase2 = Objective(expr=model.alpha, sense=maximize)

        self._solve(model)
        print("ALPHA = %.2f %%" % (model.alpha.value * 100))

        # Average results over quarter
        quarter_results = {e: None for e in self.entities}
        tmp_results = {e: [] for e in model.entities}
        for p in model.periods:
            for e in model.entities:
                tmp_results[e].append(
                    [quarter + timedelta(minutes=p - 1),
                     model.price_com[e, p].value,
                     model.exp_grid[e, p].value,
                     model.imp_grid[e, p].value,
                     model.exp_com[e, p].value,
                     model.imp_com[e, p].value])
        detailed_results = {
            e: pd.DataFrame(v, columns=["DateTime", "price_com", "exp_grid", "imp_grid", "exp_com", "imp_com"])
            for (e, v) in tmp_results.items()}

        for e in self.entities:
            quarter_results[e] = detailed_results[e].mean()
            num = (detailed_results[e]["exp_com"] - detailed_results[e]["imp_com"]) * detailed_results[e]["price_com"]
            num = num.sum()
            den = (quarter_results[e]["exp_com"] - quarter_results[e]["imp_com"])
            if not (is_zero(den)):
                quarter_results[e]["price_com"] = num / den / self.n_periods_in_quarter

            quarter_results[e] = quarter_results[e].round(decimals=5)

        # Update the past_peak of the community
        self.community_past_peak = max(self.community_past_peak, sum(model.peak_MU[e].value for e in model.entities))
        # print("com peak: ", self.community_past_peak)

        # Store entitiy profits for verification
        for e in model.entities:
            self.comparison_df[e].append([quarter, entity_alone_profits[e], model.J_MU[e].value])

        return quarter_results

    def _sale_price(self, e: int, p: int, q: pd.Timestamp) -> float:
        """
        Access to input data.
        :param e: entity
        :param p: period
        :param q: quarter start time
        :return: sale price of entity e in period p of quarter q [EUR/kWh]
        """
        entity = self.entities[e]
        return data["sale_price_%d" % entity.entity_id][self._period_to_timestamp(p, q)]

    def _period_to_timestamp(self, p: int, q: pd.Timestamp) -> pd.Timestamp:
        """
        :param p: period index within quarter q
        :param q: start time of quarter
        :return: time stamp of period
        """
        return q + timedelta(minutes=(p - 1) * self.period_duration_in_minutes)

    def _purchase_price(self, e: int, p: int, q: pd.Timestamp) -> float:
        """
        Access to input data.
        :param e: entity
        :param p: period
        :param q: quarter start time
        :return: purchase price of entity e in period p of quarter q [EUR/kWh]
        """
        entity = self.entities[e]
        return data["purchase_price_%d" % entity.entity_id][self._period_to_timestamp(p, q)]

    def _injection(self, e: int, p: int, q: pd.Timestamp) -> float:
        """
        Access to input data.
        :param e: entity
        :param p: period
        :param q: quarter start time
        :return: injection of entity e in period p of quarter q [kW]
        """
        entity = self.entities[e]
        if entity.EAN_inj is not None:
            return data[entity.EAN_inj][self._period_to_timestamp(p, q)]
        else:
            return 0.

    def _withdrawal(self, e: int, p: int, q: pd.Timestamp) -> float:
        """
        Access to input data.
        :param e: entity
        :param p: period
        :param q: quarter start time
        :return: withdrawal of entity e in period p of quarter q [kW]
        """
        entity = self.entities[e]
        if entity.EAN_wd is not None:
            return data[entity.EAN_wd][self._period_to_timestamp(p, q)]
        else:
            return 0.

    def _net_position(self, e, p, quarter):
        """
        Access to input data.
        :param e: entity
        :param p: period
        :param q: quarter start time
        :return: net position [kW] of entity e at period p within a quarter: > 0 if in export position, <0 in import position.
        """
        return self._injection(e, p, quarter) - self._withdrawal(e, p, quarter)

    def _add_sets(self, model):
        """ Utility function to add sets to a pyomo model. Used in both models (entity and community). """
        model.entities = Set(initialize=self.entities.keys())
        model.periods = RangeSet(self.n_periods_in_quarter)

    def _add_grid_vars(self, model):
        """ Utility function to add grid variables to a pyomo model. Used in both models (entity and community). """
        model.exp_grid = Var(model.entities, model.periods, within=NonNegativeReals)  # Power sent to the grid [kW]
        model.imp_grid = Var(model.entities, model.periods,
                             within=NonNegativeReals)  # Power imported from the grid [kW]

    def _solve(self, model: ConcreteModel, name: str = "community"):
        """
        Solve model and try multiple ipopt tol tolerance enlargements if problem is infeasible.
        :param model: Pyomo model
        :param name: in ["entity", "community"]
        :return: Nothing if success (model is loaded with solution), or SolverError exception in case it does not work.
        """

        assert (name in ["entity", "community"])

        results = self.opt.solve(model, tee=False, keepfiles=False)
        while not ((results.solver.status == SolverStatus.ok) and (
                results.solver.termination_condition == TerminationCondition.optimal)):

            if (results.solver.termination_condition == TerminationCondition.infeasible) and self.ipopt_tol < 1e-5:
                self.ipopt_tol *= 10
                print("Changing ipopt_tol to %f" % self.ipopt_tol)
                self.opt.options.update({"tol": self.ipopt_tol})
                results = self.opt.solve(model, tee=False, keepfiles=False)
            else:
                if name == "entity":
                    model.write(filename="entity.lp",
                                format=ProblemFormat.cpxlp,
                                io_options={"symbolic_solver_labels": True})
                else:
                    model.write(filename="community.nl")

                # Something else is wrong
                raise (SolverError(
                    "Solver Status: %s , %s" % (results.solver.status, results.solver.termination_condition)))

        # reset to default tolerances
        self.ipopt_tol = DEFAULT_IPOPT_TOL


if __name__ == "__main__":
    import json

    with open("example/invoicing.json", "r") as invoicing:
        structure = json.load(invoicing)  # Read the structure
    data = pd.read_csv("example/input_for_invoicing.csv", index_col="datetime", parse_dates=True)  # Read the data
    invoicer = Invoicer(structure, data)  # Instantiante Invoicer, default number of periods per quarter.
    invoices = invoicer.compute_invoice()  # Compute invoices

    # Plotting and reporting
    from pathlib import Path

    RESULTS_PATH = "./results/"
    Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    for e, df in invoices.items():
        # Invoice related values
        df.to_csv(RESULTS_PATH + "invoice_%d.csv" % e)
        df[["price_com"]].plot()
        plt.savefig(RESULTS_PATH + "prices_%d.png" % e, dpi=200)
        df["exp"] = df["exp_grid"] + df["exp_com"]
        df["imp"] = df["imp_grid"] + df["imp_com"]
        df[["exp_grid", "imp_grid", "exp_com", "imp_com", "exp", "imp"]].plot()
        plt.savefig(RESULTS_PATH + "flows_%d.png" % e, dpi=200)

        # Profit comparison
        df2 = pd.DataFrame(data=invoicer.comparison_df[e],
                           columns=["DateTime", "profit_alone", "profit_com"])
        df2.set_index("DateTime", inplace=True)
        df2.plot()
        df2.to_csv(RESULTS_PATH + "sol_%d.csv" % e)
        plt.savefig(RESULTS_PATH + "profits_%d.png" % e, dpi=200)
        print("Entity %d" % e, df2.sum())
        print("peak:", invoicer.entities[e].past_peak)

    print("Community PEAK:", invoicer.community_past_peak)

    # print(invoices)
