# Chris 06.03.2024
# usful functions for the lab script, this to be imported in other scripts for usuage

def linear_func(x, m, c):
    return m * x + c

def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c

# calculates the T_R from the normalised resistance
def cal_temp_from_normalised_resistance(resistance):
    return quadratic_func(resistance, -1.5992799065020011, 196.7283569841743, 139.4149679140698)

# calculates T_B from the T_R
def cal_temp_B_from_temp_R(temp_R):
    return linear_func(temp_R, 0.9558987321709527, 127.60388173355331)