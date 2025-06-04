# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# This file contains test examples used in
# tests/autoquant/cfg/test_reconstruct.py::test_reconstruct

# CASE: Function with single if statement
def my_function_1() -> None:
    if test:
        pass
    do_something()


# EXPECT: exact
# ENDCASE
# ------------------------------------------------------------------------------


# CASE: Function with if/else statement
def my_function_2() -> None:
    if test:
        pass
    else:  # a comment
        pass
    do_something()


# EXPECT: exact
# ENDCASE
# ------------------------------------------------------------------------------


# CASE: Function with if/elif statement
def my_function_3() -> None:
    if test:
        pass
    elif test2:
        pass
    do_something()


# EXPECT: exact
# ENDCASE
# ------------------------------------------------------------------------------


# CASE: Function with if/elif/else statement
def my_function_4() -> None:
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
def my_function_5() -> None:
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
def my_function_6() -> None:
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


# CASE: Function with for loop
def my_function_7() -> None:
    for x in range(10):
        print(x)


# EXPECT: exact
# ENDCASE
# ------------------------------------------------------------------------------


# CASE: Function with while loop
def my_function_8() -> None:
    while do_something():
        print("iteration")


# EXPECT: exact
# ENDCASE
# ------------------------------------------------------------------------------


# CASE: Function with with statement
def my_function_9() -> None:
    filename = "my_file"
    with open(filename) as f:
        f.write("text")
    print("finalize")


# EXPECT: exact
# ENDCASE
# ------------------------------------------------------------------------------
#
# CASE: Function with nested with statements
def my_function_10() -> None:
    filename1 = "my_file1"
    filename2 = "my_file2"
    filename3 = "my_file3"
    with open(filename1) as f1:
        f1.write("text1")
        with open(filename2) as f2:
            f2.write("text2")
            with open(filename3) as f3:
                f3.write("text3")
    print("finalize")


# EXPECT: exact
# ENDCASE
# ------------------------------------------------------------------------------
#
# EXPECT: exact
# ENDCASE
# ------------------------------------------------------------------------------
#
# CASE: Function with two nested with statements
def my_function_11() -> None:
    filename1 = "my_file1"
    filename2 = "my_file2"
    filename3 = "my_file3"
    with open(filename1) as f1:
        f1.write("text1")
        with open(filename2) as f2:
            f2.write("text2")
        value = 100 * 2
        with open(filename3):
            print("do nothing")
        print(value)
    print("finalize")


# EXPECT: exact
# ENDCASE
# ------------------------------------------------------------------------------

# Define symbols used in test functions to suppress errors
test = True
test2 = False


def do_something() -> bool:
    return True


def do_more() -> None: ...
