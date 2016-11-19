import solver
import sys
import printer

def main():
    if len(sys.argv) < 2:
        print "Usage: python main.py [spec file] <spec files>"
        exit(0)
    s = solver.Solver()
    for spec in sys.argv[1:]:
        s.solve(spec)

if __name__ == '__main__':
    main()
