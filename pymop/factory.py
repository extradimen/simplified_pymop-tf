from pymop import *

STR_TO_PROBLEM = {
    'dtlz1': DTLZ1,
    'c1dtlz1': C1DTLZ1,
}


def get_problem(name, *args, **kwargs):
    return STR_TO_PROBLEM[name.lower()](*args, **kwargs)

