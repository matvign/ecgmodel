#!/usr/bin/env python3
def triway(lst):
    it = iter(lst)
    while True:
        try:
            yield next(it), next(it), next(it)
        except StopIteration:
            return