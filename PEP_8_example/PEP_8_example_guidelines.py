# Purpose of PEP8 guidelines:
# To improve consistency and readability of code within and across teams

# Imports 
# start with standard libraries
import math
import sys
import os 
# do not put more than one on a line with a comma

# then import external 
import numpy as np
import pandas as pd

# last is local
from mymodule import myClass, otherClass
# okay to use commas for local imports


# Length
# Keep under 79 characters per line - so this example on this line is way too long and needs to be multiple lines...


# Indentations 
# part 1: functions
def function_with_multiple_variable_inputs(
        one_in, two_in, three_in, four_in, 
        five_in, six_in, seven_in. eight_in):
    print("Done")

# part 2: if statements
one_boolean_thing, another_boolean_thing = True, True

if (one_boolean_thing 
        and another_boolean_thing):
    print("Done")

# part 3: lists
long_list = [
    0, 1, 2,
    3, 4, 5
    ]

# part 4: breaking up operators
income = (gross_wages 
          + taxable_interest
          + (dividends - qualified_dividens)
          - ira_deduction)
# keep the operators with the operands

# Tabs/Spaces
def tabs_or_spaces():
    tabs = 1
    spaces = 4
# either works, keep it consistent and don't mix


# Single or double-quotes
single = 'single quotes'
double = "double quotes"
# either works, keep it consistent and don't mix


# Whitespaces 
# after comas
x, y, z = 1, 2, 3

# around signs
a = 1
b = 2
abcdefg = 3

# omit in functions
def function(one_value, two_value=2.0):
    return one_value + two_value

# omit when calling function
function(one_value=3, two_value=3)

# ommit trailing common for a close parenthesis
d = (0,)