from itertools import chain

def parse_range(rng):
    parts = rng.split('-')
    if 1 > len(parts) > 2:
        raise ValueError("Bad range: '%s'" % (rng,))
    parts = [int(i) for i in parts]
    start = parts[0]
    end = start if len(parts) == 1 else parts[1]
    if start > end:
        end, start = start, end
    return range(start, end + 1)

def parse_range_list(rngs):
    """Convert String ranges to list

    Arguments:
    rngs -- {str} -- example: "1, 4-6, 8-10, 11"

    Returns:
    List after conversion from string : [1, 4, 5, 6, 8, 9, 10, 11]
    """
    return sorted(set(chain(*[parse_range(rng) for rng in rngs.split(',')])))

def isnotebook():
    """Check if code is executed in the IPython notebook
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
