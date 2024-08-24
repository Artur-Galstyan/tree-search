def counter():
    idx = 0

    def increment():
        nonlocal idx
        idx += 1
        return idx - 1

    return increment
