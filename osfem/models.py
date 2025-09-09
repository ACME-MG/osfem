"""
 Title:         Models
 Description:   Simple empirical models
 Author:        Janzen Choi

"""

# TTF, Monkman-Grant
def ttf_mg(data, c, m):
    stf, mcr = data["stf"], data["mcr"]
    ttf = stf*(c/mcr)**m
    return ttf

# STF, Evans and Wilshire
def stf_ew(data, a, b, c, d):
    s, t = data["stress"], data["temperature"]
    stf = a + b*s + c*t + d*s*t
    return stf
