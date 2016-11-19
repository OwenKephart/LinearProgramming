from scipy.optimize import linprog
import numpy as np
import parser
import printer

"""
This class takes the parsed expressions and turns them into a linear
program. Makes no implicit assumptions about the goals of the user.
"""
class Solver:


    """
    parse the requirements file and generate numpy arrays to use as input
    to the solver, then solve the program
    """
    def solve(self, filename, force_ilp=False):
        # parse the file
        p = parser.Parser()
        reqs, goals = p.parse_file(filename)

        
        printer.print_start_process("Optimizing Program")
        printer.print_update("Formulating Problem")
        A_eq, b_eq, A_ub, b_ub, c = self.formulate_LP(reqs, goals)

        printer.print_update("Optimizing")
        res = linprog(c, A_ub, b_ub, A_eq, b_eq)

        printer.print_end_process("Optimizing Program")
        printer.print_result(p, res)

    """
    formulate as an ILP
    """
    def formulate_ILP(self, reqs, goals):
        pass

    """
    formulate as a regular LP
    """
    def formulate_LP(self, reqs, goals):

        num_cats = len(goals[0][1])
        num_goals = len(goals)

        # initialize with no weights
        c = [0]*(num_cats + 2*num_goals)
        A_eq = []
        b_eq = []
        A_ub = []
        b_ub = []

        # empty arrays as padding for requirements
        goal_weights = [0]*(2*num_goals)
        for req in reqs:
            # for equality, punish both positive and negative deviation
            if req[0] == parser.Parser.EQUAL:
                req_lhs = req[1]
                req_rhs = req[2]
                A_eq.append(req_lhs + goal_weights)
                b_eq.append(req_rhs)
            elif req[0] == parser.Parser.LESS:
                req_lhs = req[1]
                req_rhs = req[2]
                A_ub.append(req_lhs + goal_weights)
                b_ub.append(req_rhs)
            # negate greater weights
            elif req[0] == parser.Parser.GREATER:
                req_lhs = [-x for x in req[1]]
                req_rhs = -req[2]
                A_ub.append(req_lhs + goal_weights)
                b_ub.append(req_rhs)

        # generate the positive and negative auxillary variables for goals
        # all constraints are equality constraints, because these are 
        # assignments
        for (i, goal) in enumerate(goals):
            # just generate a progressively smaller number for the weights
            cur_obj_weight = self.BIG_M**(len(goals)-i-1)
            cur_obj_ind = num_cats + 2*i
            goal_weights = [0, 0]*(i) + [-1, 1] + [0, 0]*(num_goals-i-1)
            # choose the weights in the objective function
            if goal[0] == parser.Parser.EQUAL:
                c[cur_obj_ind] = cur_obj_weight # minimize either deviation
                c[cur_obj_ind+1] = cur_obj_weight
                goal_lhs = goal[1]
                goal_rhs = goal[2]
            elif goal[0] == parser.Parser.LESS:
                c[cur_obj_ind] = cur_obj_weight # only minimize positive deviation 
                c[cur_obj_ind+1] = 0
                goal_lhs = goal[1]
                goal_rhs = goal[2]
            elif goal[0] == parser.Parser.GREATER:
                c[cur_obj_ind] = 0  
                c[cur_obj_ind+1] = cur_obj_weight # only minimize negative deviation
                goal_lhs = [-x for x in goal[1]]
                goal_rhs = -goal[2]
            A_eq.append(goal_lhs + goal_weights)
            b_eq.append(goal_rhs)

        # convert to numpy
        A_eq = np.array(A_eq) 
        b_eq = np.array(b_eq) 
        A_ub = np.array(A_ub) 
        b_ub = np.array(b_ub) 
        c = np.array(c)

        return (A_eq, b_eq, A_ub, b_ub, c)

    # progressive weighting factor for the goals (should just be pretty large)
    BIG_M = 10
