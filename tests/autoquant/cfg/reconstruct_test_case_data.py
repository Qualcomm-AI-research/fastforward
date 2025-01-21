# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# This file contains test examples used in
# tests/autoquant/cfg/test_reconstruct.py::test_reconstruct

# CASE: Function with single if statement
def my_function_1():
    if test:
        pass
    do_something()


# EXPECT: exact
# ENDCASE
# ------------------------------------------------------------------------------


# CASE: Function with if/else statement
def my_function_2():
    if test:
        pass
    else:  # a comment
        pass
    do_something()


# EXPECT: exact
# ENDCASE
# ------------------------------------------------------------------------------


# CASE: Function with if/elif statement
def my_function_3():
    if test:
        pass
    elif test2:
        pass
    do_something()


# EXPECT: exact
# ENDCASE
# ------------------------------------------------------------------------------


# CASE: Function with if/elif/else statement
def my_function_4():
    if test:
        pass
    elif test2:
        pass
    else:
        pass
    do_something()


# EXPECT: exact
# ENDCASE
# ------------------------------------------------------------------------------


# CASE: Function with if/else with single if in else block
def my_function_5():
    if test:
        pass
    else:  # a comment
        if test2:
            pass
    do_something()


# EXPECT: exact
# ENDCASE
# ------------------------------------------------------------------------------


# CASE: Function with if statement in clause with a trailing statement
def my_function_6():
    if test:
        pass
    else:  # a comment
        if test2:
            pass
        do_more()
    do_something()


# EXPECT: exact
# ENDCASE
# ------------------------------------------------------------------------------

# Define symbols used in test functions to suppress errors
test = True
test2 = False


def do_something(): ...
def do_more(): ...
