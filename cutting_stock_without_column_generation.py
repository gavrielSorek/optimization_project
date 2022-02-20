import cvxopt
import numpy as np
from cvxopt.glpk import ilp
import time
from cvxopt import matrix


# check if pattern is valid
def is_valid_pattern(current_c, type_sizes, roll_size):
    total_size = 0
    for j in range(len(current_c)):
        total_size += current_c[j] * type_sizes[j]
        if total_size > roll_size:
            return False
    return True


# return all valid patterns
def get_all_patterns(roll_length, type_sizes):
    roll_types = len(type_sizes)
    # initialize pattern
    all_patterns = np.array([np.zeros(shape=roll_types)])
    current_pattern = np.zeros(roll_types)
    is_exist_more_valid_pattern = True
    while is_exist_more_valid_pattern:
        if is_valid_pattern(current_pattern, type_sizes, roll_length):
            to_add = np.array([current_pattern])
            all_patterns = np.append(to_add, all_patterns, axis=0)
            current_pattern[0] += 1
        else:
            current_position = 1
            while not is_valid_pattern(current_pattern, type_sizes, roll_length):
                if current_position >= len(current_pattern):
                    is_exist_more_valid_pattern = False
                    break
                current_pattern[current_position - 1] = 0
                current_pattern[current_position] += 1
                current_position += 1
    return np.delete(all_patterns, len(all_patterns) - 1, axis=0)


def initial_solution(roll_length, cut_width):
    return np.transpose(get_all_patterns(roll_length, cut_width), axes=None)


class cutting_stock_solver:
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

    def solve(self):
        initial_pattern = initial_solution(self.roll_length, self.cut_width)
        cut_pattern = initial_pattern

        minimal_stock, optimal_number = self.int_master_problem(cut_pattern)
        return minimal_stock, optimal_number, cut_pattern

# return relevant patterns from all patterns
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
        solver = cutting_stock_solver(roll_length, user_demand[:i + 1], cut_width[:i + 1])
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
