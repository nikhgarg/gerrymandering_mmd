def add_two_dicts(d1, d2, default=0):
    return {
        x: d1.get(x, default) + d2.get(x, default)
        for x in list(d1.keys()) + list(d2.keys())
    }
