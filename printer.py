import sys

def printflush(msg):
    print msg
    sys.stdout.flush()
    
# outputs the result of the LP in a nice format
def print_result(parser, res):
    if res.status != 0:
        printflush(res.message)
        printflush("Something went wrong in the solver, terminating...")
        exit(0)
    printflush("Optimal solution found!")
    printflush("---------------")
    for i in range(len(parser.all_pats)):
        mem_name = ""
        for cat, mem in parser.all_pats[i]:
            mem_name += cat + "=" + mem + ", "
        mem_name = mem_name.rstrip(", ")
        printflush(mem_name + ": " + str(res.x[i]))
    printflush("---------------")

def print_start_process(process):
    printflush(process + "...")

def print_update(action):
    printflush("\t>" + action)

def print_end_process(process):
    printflush("Finished " + process + "!")
