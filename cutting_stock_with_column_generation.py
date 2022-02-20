import cvxopt
import gurobipy as grb
from gurobipy import GRB
import numpy as np
from scipy import optimize
import time
from cvxopt.glpk import ilp
from cvxopt import matrix


def initial_solution(W, w):
    return np.diag(np.floor(W / w))


class cutting_stock_solver_column_generation:
    def __init__(self, roll_length, user_demand, cut_width):
        self.roll_length = roll_length
        self.user_demand = user_demand
        self.cut_width = cut_width

    # USING BRANCH AND BOUND TO SOLVE THIS ILP
    def int_master_problem(self, column):
        # all variable need to be positive
        positive_constrain = np.diag(np.ones(column.shape[1], dtype=float))
        right_side_positive_constrain = np.zeros(column.shape[1], dtype=float)

        c = matrix(np.ones(column.shape[1], dtype=float))
        constrains = np.append(column, positive_constrain, axis=0)
        right_side = np.append(self.user_demand, right_side_positive_constrain, axis=0)
        G = matrix(-constrains)
        h = matrix(-right_side)

        I = set(range(column.shape[1]))
        B = set()
        cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'
        (status, x) = ilp(c=c, G=G, h=h, I=I, B=B)
        results = np.zeros(len(x))
        for j in range(len(x)):
            results[j] = x[j]
        return results, sum(c.T * x)

    # USING SIMPLEX TO SOLVE THE DUAL PROBLEM
    def dual_solution(self, column):
        c = np.full(column.shape[1], 1)
        A = np.array(-1 * column)
        b = np.array(-1 * self.user_demand)
        res = optimize.linprog(b, -1 * np.transpose(A), c, method='simplex', options={"disp": False})
        return res.x

    # USING BRANCH AND CUT TO SOLVE THIS ILP
    def subproblem(self, dual_values):
        with grb.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with grb.Model(env=env) as model:
                x = model.addMVar(shape=dual_values.shape[0], lb=0, vtype=GRB.INTEGER)
                model.addConstr(lhs=self.cut_width @ x <= self.roll_length)
                model.setObjective(1 - dual_values @ x, GRB.MINIMIZE)
                model.optimize()
                is_new_col = model.objVal < 0
                if is_new_col:
                    new_column = model.getAttr('X')
                else:
                    new_column = None
                return is_new_col, new_column

    # solve the cutting stocks problem
    def solve(self):
        cut_pattern = initial_solution(self.roll_length, self.cut_width)
        is_exist_new_pattern = True
        new_pattern = None

        while is_exist_new_pattern:
            if new_pattern:
                cut_pattern = np.column_stack((cut_pattern, new_pattern))

            dual_values = self.dual_solution(cut_pattern)
            is_exist_new_pattern, new_pattern = self.subproblem(dual_values)

        minimal_stock, optimal_number = self.int_master_problem(cut_pattern)
        return minimal_stock, optimal_number, cut_pattern


def get_relevant_patterns(cut_pattern, minimal_stock):
    cut_pattern = np.transpose(cut_pattern, axes=None)
    first_cut_flag = True
    relevant_cut_patterns = None
    relevant_patterns_amount = np.array([])
    for i in range(len(minimal_stock)):
        if minimal_stock[i] != 0:
            # add the pattern
            if first_cut_flag:
                relevant_cut_patterns = np.array([cut_pattern[i]])
                first_cut_flag = False
            else:
                relevant_cut_patterns = np.append(np.array([cut_pattern[i]]), relevant_cut_patterns, axis=0)
            relevant_patterns_amount = np.append(relevant_patterns_amount, minimal_stock[i])
    return relevant_cut_patterns.astype(int), relevant_patterns_amount


def main():
    save_run_time = []
    save_amount = []
    save_patterns = []
    save_minimal_stocks = []
    save_optimal_number = []

    roll_length = np.array(60)
    cut_width = np.array([3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16])  # width
    user_demand = np.array([25, 20, 18, 10, 4, 5, 6, 7, 8, 5, 6])

    for i in range(len(user_demand)):
        start = time.time()
        solver = cutting_stock_solver_column_generation(roll_length, user_demand[:i + 1], cut_width[:i + 1])
        minimal_stock, optimal_number, cut_pattern = solver.solve()
        end = time.time()
        save_optimal_number.append(optimal_number)
        relevant_cut_patterns, relevant_patterns_amount = get_relevant_patterns(cut_pattern, minimal_stock)
        save_patterns.append(relevant_cut_patterns)
        save_minimal_stocks.append(relevant_patterns_amount)
        save_run_time.append(end - start)
        save_amount.append(i + 1)

    print("**********************************************************************************************************")
    for i in range(len(save_run_time)):
        print('                 USER PARAMETERS')
        print(f'roll length:  {roll_length} , cut width: {cut_width[:i + 1]} ,user demand: {user_demand[:i + 1]}')
        print('                 RESULT')
        print(f'Min Stocks: {save_optimal_number[i]}')
        print("amount to take from pattern X pattern")
        for j in range(len(save_minimal_stocks[i])):
            print(f'{int(save_minimal_stocks[i][j])} X {save_patterns[i][j]}')
        print("Outcome : user demand:" + str(save_amount[i]) + " values, run time is : " + str(save_run_time[i]))


main()
