from scipy.optimize import linprog
import re


class Solver:

    def __init__(self):
        pass


    def create_LP(self):
        c = [] #coeffs of objective func
        A_ub = [] # 2D matrix for upper bound consts
        b_ub = [] # 1D vector for upper bound rhs
        A_eq = [] # 2D matrix for eq consts
        b_eq = [] # 1D vector for eq rhs
        bounds = [(0,None)] # for the x vector
        pass


class Parser:


    def __init__(self):
        pass

    # get the categories, applicants, goals, and requirements
    # make sure that it is formatted correctly
    def parse_file(self, fname):
        # get cleaned up list of lines in file with line numbers
        contents = self.strip_contents(open(fname).readlines())
        self.print_data(contents)

        # get the set of lines where each header occurs
        header_locs = self.find_headers(contents)

        # get the set of categories
        cats = self.parse_cats(
                self.get_slice(contents, header_locs.get(self.CAT_H)))
        # all the possible combinations of members
        #member_combos = self.get_member_combos(cats)

        # get the input values
        vals = self.parse_vals(cats,
                self.get_slice(contents, header_locs.get(self.VAL_H)))

        # get the requirement and goal data
        reqs = self.parse_constraints(cats, vals,
                self.get_slice(contents, header_locs.get(self.REQ_H)))

        goals = self.parse_constraints(cats, vals,
                self.get_slice(contents, header_locs.get(self.GOAL_H)))

        print(reqs)
        print(goals)

    # useful for debugging
    def print_data(self, data):
        print("--------------------------")
        for (new_num, (orig_num, line)) in enumerate(data):
            print(str(orig_num) + ":\t" + str(new_num) + "\t"  + line)
        print("--------------------------")

    # parse the categories into a list of (name, list)
    def parse_cats(self, data):
        cats = []
        for el in data:
            # split on the divider character (should be ":")
            (cat, members_str) = self.divide(el)
            # this happens if there was a syntax error
            if cat is None:
                continue
            # turn the string into a list
            members = members_str.split(self.SEP_CHAR)
            cats.append((cat, members))
        return cats

    # parse the input values for the categories into a dict pointing to ints
    def parse_vals(self, cats, data):
        
        vals = {}
        for el in data:
            # split on te divide character (should be ":") 
            (lhs, rhs) = self.divide(el)
            # this happens if there was a syntax error
            if lhs is None:
                continue

            # parse the pattern
            # (i.e. "in[gender=male,class_year=1] -> [(gender,male),(class_year,1)]
            pattern = self.get_pattern(lhs)
            # make sure this is a valid thing (i.e. not in[(zzx, 1)])
            if not self.check_pattern(cats, pattern, req_all=True):
                self.error(el[0], "Invalid membership: \"" + lhs + "\"")

            pattern_str = self.pattern_to_string(cats, pattern)
            if pattern_str in vals:
                self.error(el[0], "Duplicate definition: \"" + lhs + "\"")
                continue

            try:
                vals[pattern_str] = int(rhs)
            except:
                self.error(orig_num, "RHS must be a number: \"" + rhs + "\"")

        return vals

    # parses a set of constraints that can be either numeric or percentage
    def parse_constraints(self, cats, vals, data):

        consts = []
        for el in data:
            # split on the divider character (should be ":")
            (kind, ineq) = self.divide(el)
            # this happens if there was a syntax error
            if kind is None:
                continue
            if kind == "numeric":
                consts.append(self.parse_ineq_numeric(cats, vals, ineq))
            
        return consts

    # parses an inequality
    def parse_ineq_numeric(self, cats, vals, ineq):

        m_op = None
        for op in self.OPS:
            if op in ineq:
                m_op = op
                sp = ineq.split(m_op)
                (lhs, rhs) = self.divide((-1,ineq),div=m_op)
                if lhs is None:
                    self.error(-1, "Invalid expression: \"" + ineq + "\"")
        if m_op is None:
            self.error(-1, "Invalid operation: \"" + ineq + "\"")

        lhs = self.parse_lhs(cats,vals,lhs)
        rhs = self.parse_rhs(cats,vals,rhs)
        return (m_op, lhs, rhs)

    # look for all the output variables and their coefficients
    # return a list of coefficients and associated output variables 
    def parse_lhs(self, cats, vals, s):
        # regex magic
        p = re.compile("(?:\d*\.\d+\*)?out\[(?:\w+=\w+,*)*\]")
        # parse individual expressions
        exprs = [self.parse_expr(x) for x in p.findall(s)]

        # expand out expressions
        full_exprs = []
        for expr in exprs:
            if not self.check_pattern(cats,expr[1]): # make sure valid
                self.error(-1, "Invalid membership: \"" + s + "\"")
            matches = self.get_patterns_matching(cats, expr[1])
            for match in matches:
                full_exprs.append((expr[0], match))
        return full_exprs

    # look for all the input variables and their coefficients
    # return the numerical sum
    # (the regex works, even if it's hard to read)
    def parse_rhs(self, cats, vals, s):
        # regex magic
        p = re.compile("(?:\d*\.\d+\*)?in\[(?:\w+=\w+,*)*\]|(?:\d*\.?\d+)")
        # parse individual expressions
        exprs = [self.parse_expr(x) for x in p.findall(s)]

        # get numeric value for rhs
        for expr in exprs:
            if expr[1] is not None and not self.check_pattern(cats,expr[1]):
                self.error(-1, "Invalid membership: \"" + s + "\"")

        # multiply and add
        val = 0
        for expr in exprs:
            if expr[1] is None: # plain number value
                val += expr[0]
            else: # variable number value
                val += expr[0]*self.get_val(cats,vals,expr[1])
        return val

    # parses an expression in the form of .75*in[...] into (.75, ...)
    def parse_expr(self, s):
        sp = s.split('*')
        if len(sp) == 1: # either just a pattern or just a number
            try:
                f = float(sp[0])
                return (f, None)
            except:
                return (1.0, self.get_pattern(sp[0]))
        elif len(sp) == 2:
            return (float(sp[0]), self.get_pattern(sp[1]))
        else:
            self.error(-1, "Illegal expression: \"" + s + "\"")

    # will return the sum of all values matching the pattern
    def get_val(self, cats, vals, pattern):
        all_matches = self.get_patterns_matching(cats, pattern)
        queries = [self.pattern_to_string(cats, x) for x in all_matches]
        vals = [vals[x] for x in queries]
        return sum(vals)

    # turn a string into a pattern (it works, but it doesn't detect errors
    # too well - usually they will be caught later though)
    def get_pattern(self, s):
        s = s[s.find(self.OPEN_CHAR)+1:s.find(self.CLOSE_CHAR)].strip(self.SEP_CHAR)
        pattern = [x.split(self.EQ_CHAR) for x in s.split(self.SEP_CHAR)]
        return pattern

    # turn a pattern into an ordered string used for hashing
    def pattern_to_string(self, cats, pattern):

        s = self.OPEN_CHAR
        for (cat_name, _) in cats:
            for (cat, mem) in pattern:
                if cat_name == cat:
                    s += cat + self.EQ_CHAR + mem + self.SEP_CHAR
        return s.rstrip(self.SEP_CHAR) + self.CLOSE_CHAR


    # return if this is a valid pattern based on the possible categories and
    # member values. optionally require all categories to be represented
    def check_pattern(self, cats, pattern, req_all=False):

        # make sure the pattern has enough categories
        if req_all and (len(pattern) != len(cats)):
            return False

        cat_names = [x[0] for x in cats]
        mem_names = [x[1] for x in cats]

        cs = [] # categories that have been seen so far
        for cat, mem in pattern:
            if cat in cs: # duplicate cat name
                return False
            if cat not in cat_names:
                return False # invalid cat name
            cat_idx = cat_names.index(cat)
            if mem not in mem_names[cat_idx]: # invalid member name
                return False
            cs.append(cat)
        return True

    # takes pattern of the form "[(gender,male),(class_year,4)]", returns set
    # of things that match this pattern
    def get_patterns_matching(self, cats, pattern):

        cat_names_pat = [x[0] for x in pattern]
        mem_names_pat = [x[1] for x in pattern]

        cat_names_rst = [x[0] for x in cats]
        mem_names_rst = [x[1] for x in cats]

        # remove categories that have already been accounted for in the pattern
        for cat_name in cat_names_pat:
            cat_idx = cat_names_rst.index(cat_name)
            cat_names_rst.pop(cat_idx) 
            mem_names_rst.pop(cat_idx) 

        # at each step, add on all members to all matches
        matches = [pattern]
        for (i, cat_name) in enumerate(cat_names_rst):
            new_matches = []
            for match in matches:
                for mem in mem_names_rst[i]:
                    new_matches.append(match + [(cat_name,mem)])
            matches = new_matches

        return matches


    # prints out errors with references to original line numbers
    # will quit if necessary
    def error(self, line_num, text="Undefined Error", force_quit=True):
        print("Error on line " + str(line_num) + ": " + text)
        if force_quit:
            print("Error was fatal, quitting...")
            exit(0)

    # divide a string into exactly two parts
    def divide(self, el, div=":"):
        (orig_num, line) = el
        sp = line.split(div)
        if len(sp) != 2:
            self.error(orig_num, "Syntax Error")
            return (None, None)
        return sp

    # gets a slice of the annotated lines array
    def get_slice(self, contents, loc):
        return contents[loc[0]:loc[1]]

    # remove all comments, whitespace, and empty lines,
    # annotate with line numbers
    def strip_contents(self, lines):
        ret = [] # new list of lines
        for (num, line) in enumerate(lines):
            # remove everything after the comment symbol
            if self.COMMENT_CHAR in line:
                line = line[:line.index(self.COMMENT_CHAR)]
            # remove whitespace
            line = "".join(line.split())
            # don't add in empty lines
            if line != "":
                ret.append((num, line))
        return ret

    # get the start and end locations of each header, return as a dict
    # does error checking along the way 
    def find_headers(self, contents):
        
        # find all locations of headers in the file
        header_locs = {}
        
        # make sure first line is a header
        if not contents[0][1].startswith(self.HEADER_CHAR):
            self.error(contents[0][0], "Must start with header")

        cur_header = None
        cur_range = [None, None]
        for (new_num, (orig_num, line)) in enumerate(contents):
            if line.startswith(self.HEADER_CHAR):
                h = line[1:] # get the header
                if h not in self.HEADERS: 
                    self.error(orig_num, "Illegal header: \"" + h + "\"")
                else:
                    # start of range is 1 past the header
                    if cur_range[0] is None:
                        cur_header = h
                        cur_range[0] = new_num + 1
                    # end of the range is the next header
                    elif cur_range[1] is None:
                        cur_range[1] = new_num
                        header_locs[cur_header] = cur_range
                        
                        cur_header = h
                        if h in header_locs:
                            self.error(orig_num, "Duplicate header: \"" + h + "\"")
                        cur_range = [cur_range[1] + 1, None]
                    # should not happen, but if it does, exit
                    else:
                        self.error(orig_num, force_quit=True)
        header_locs[cur_header] = [cur_range[0], len(contents)]

        return header_locs


    # character that will begin a comment
    COMMENT_CHAR = '#'
    # character that will begin a header
    HEADER_CHAR = '+'
    # character that divides labels from data
    DIV_CHAR = ':'
    # character that separates list elements
    SEP_CHAR = ','
    # you get the idea
    OPEN_CHAR = "["
    CLOSE_CHAR = "]"
    EQ_CHAR = "="

    # header string for requirements
    REQ_H = 'requirements'
    # header string for goals
    GOAL_H = 'goals'
    # header string for categories
    CAT_H = 'categories'
    # header string for applicants
    VAL_H = 'applicants'


    # list of all valid headers
    HEADERS = [REQ_H, GOAL_H, CAT_H, VAL_H]

    # list of all valid operations for constraints
    LESS = "<="
    GREATER = ">="
    EQUAL = "=="
    OPS = [LESS,GREATER,EQUAL]


p = Parser()
p.parse_file("reqs.txt")





