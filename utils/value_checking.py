def check_boundary_value(name, val, min_val=None, max_val=None):
    if min_val and val < min_val:
        raise ValueError("{}: Invalid value {}. Value must be greater than {}".format(name, val, min_val))
    if min_val and val > max_val:
        raise ValueError("{}: Invalid value {}. Value must be smaller than {}".format(name, val, min_val))
    return val


def check_option_value(name, val, options):
    if val not in options:
        raise ValueError("{}: Invalid value {}. Possible options: {}".format(name, val, options))
    return val
