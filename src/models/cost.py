import math
import numpy as np
import pandas as pd

from typing import Tuple


# Approximate average inpatient cost per day (adjusted for inflation)
# https://www.kff.org/health-costs/state-indicator/expenses-per-inpatient-day/?currentTimeframe=0&sortModel=%7B%22colId%22:%22Location%22,%22sort%22:%22asc%22%7D
AVERAGE_INPATIENT_COST_PER_DAY = 2827.

# Approximate average patient readmission cost (adjusted for inflation)
# https://www.hcup-us.ahrq.gov/reports/statbriefs/sb248-Hospital-Readmissions-2010-2016.jsp
AVERAGE_READMISSION_COST = 15556.


# Average number of hours gerontologic nursing specialists spent on the comprehensive discharge planning methodology described in https://doi.org/10.7326/0003-4819-120-12-199406150-00005
CDP__AVERAGE_NUM_HOURS_FOR_NURSING_SPECIALIST = math.ceil(4.363)

# Reengineered Discharge Strategy https://doi.org/10.7326/0003-4819-150-3-200902030-00007
# Estimated average number of hours worked per intervention by the discharge advocate
RED__AVERAGE_NUM_HOURS_FOR_DA = math.ceil(87.5 / 60)

# Estimated average hourly wage for a discharge advocate
RED__AVERAGE_HOURLY_WAGE_FOR_DA = 50.

# Estimated average number of hours worked per intervention by the pharamacist who is responsible for the telephonic component of the process
RED__AVERAGE_NUM_HOURS_FOR_PHARMACIST = math.ceil(26 / 60)

# Estimated average hourly wage for a pharmacist
RED__AVERAGE_HOURLY_WAGE_FOR_PHARMACIST = 70.


def compute_total_readmission_cost(y_true: pd.Series, y_pred: pd.Series,
                                   estimated_readmission_cost: np.float64=AVERAGE_READMISSION_COST) -> np.float64:
    summed = y_true + y_pred
    num_false_negatives = len(summed[(summed == 1) & (y_true == 1)])
    return num_false_negatives * estimated_readmission_cost


def compute_average_operational_cost(y_true: pd.Series, y_pred: pd.Series,
                                     estimated_intervention_cost: np.float64,
                                     estimated_readmission_cost: np.float64) -> np.float64:
    if len(y_true) != len(y_pred):
        raise ValueError(f'Given truth and prediction series have mismatched lengths: {len(y_true)} != {len(y_pred)}')

    summed = y_true + y_pred
    
    num_false_positives = len(summed[(summed == 1) & (y_true == 0)])
    num_false_negatives = len(summed[(summed == 1) & (y_true == 1)])
    num_true_positives  = len(summed[summed == 2])

    cost = (num_true_positives * (estimated_intervention_cost - estimated_readmission_cost)) + \
           (num_false_negatives * estimated_readmission_cost) + \
           (num_false_positives * estimated_intervention_cost)

    return cost / len(y_true)


def compute_operational_cost_with_longer_stay(y_true: pd.Series, y_pred: pd.Series,
                                              average_inpatient_cost_per_day: np.float64=AVERAGE_INPATIENT_COST_PER_DAY,
                                              average_readmission_cost: np.float64=AVERAGE_READMISSION_COST) -> np.float64:
    return compute_average_operational_cost(y_true, y_pred, estimated_intervention_cost=average_inpatient_cost_per_day, estimated_readmission_cost=average_readmission_cost)


def compute_average_operational_cost_with_discharge_planning(y_true: pd.Series, y_pred: pd.Series,
                                                             nursing_specialist_estimated_hourly_wage: np.float64=65.,
                                                             nursing_specialist_average_num_hours_worked: np.float64=CDP__AVERAGE_NUM_HOURS_FOR_NURSING_SPECIALIST,
                                                             average_readmission_cost: np.float64=AVERAGE_READMISSION_COST) -> np.float64:
    return compute_average_operational_cost(y_true, y_pred, estimated_intervention_cost=nursing_specialist_average_num_hours_worked*nursing_specialist_estimated_hourly_wage, estimated_readmission_cost=average_readmission_cost)

def compute_average_operational_cost_with_red(y_true: pd.Series, y_pred: pd.Series,
                                              discharge_advocate_estimated_hourly_wage: np.float64=RED__AVERAGE_HOURLY_WAGE_FOR_DA,
                                              discharge_advocate_average_num_hours_worked_per_intervention: np.float64=RED__AVERAGE_NUM_HOURS_FOR_DA,
                                              pharmacist_estimated_hourly_wage: np.float64=RED__AVERAGE_HOURLY_WAGE_FOR_PHARMACIST,
                                              pharmacist_average_num_hours_worked_per_intervention: np.float64=RED__AVERAGE_NUM_HOURS_FOR_PHARMACIST,
                                              average_readmission_cost: np.float64=AVERAGE_READMISSION_COST) -> np.float64:
    
    return compute_average_operational_cost(y_true, y_pred,
                                            estimated_intervention_cost=70. + discharge_advocate_estimated_hourly_wage * discharge_advocate_average_num_hours_worked_per_intervention + pharmacist_estimated_hourly_wage * pharmacist_average_num_hours_worked_per_intervention,
                                            estimated_readmission_cost=average_readmission_cost)
