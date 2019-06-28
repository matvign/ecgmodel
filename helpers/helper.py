#!/usr/bin/env python3
import re

pi_regx = re.compile(r"(pi\s*|[0-9.]+)(?=(\s*pi|[0-9.]))")


def triway(lst):
    it = iter(lst)
    while True:
        try:
            yield next(it), next(it), next(it)
        except StopIteration:
            return


def pirepl(word):
    def repl(matchobj):
        if matchobj.group(1).strip() == 'pi':
            return "{}*".format(matchobj.group(1))
        elif matchobj.group(2).strip() == 'pi':
            return "{}*".format(matchobj.group(1))
        else:
            return matchobj.group(0)

    return re.sub(pi_regx, repl, word)