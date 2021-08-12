#!/usr/bin/env python
# -*- coding: utf-8 -*-


def activate_functions_mapper(func_names, module):
    if not type(func_names) is list:
        func_names = [func_names]

    try:
        result = [getattr(module, name) for name in func_names]
        if len(result) == 1:
            return result[0]
        else:
            return result
    except AttributeError as e:
        raise AttributeError("{}".format(e))