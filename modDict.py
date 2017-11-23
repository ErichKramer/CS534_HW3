from collections import defaultdict
from copy import deepcopy

class myclass(defaultdict):
    def __add__(self, other):
        tmp = deepcopy(self) #this makes code slow but robust. 
        for key in other.keys():
            try:
                tmp[key] += other[key]
            except TypeError:#put a square peg in a round hole anyway
                tmp[key] = ( tmp[key], other[key])
                
                if debug:
                    print("#! ERROR: MALFORMED DICT SQUASH. TYPES NOT EQUAL !#", \
                            file=__import__("sys").stderr)
                pass
        return tmp



if __name__ == "__main__":
    debug = True
    a = myclass(int)
    b = myclass(int)
    a[0] = a[1] + a[2]
    b["yes"] = 100
    print(a+b)

    c = myclass(int)
    d = myclass( lambda : "Who would even use a string as a default")

    c[0] = "hello world"
    c[1] = 5
    d[100] = "types are really just a recommendation"
    print(c+d)
    pass
