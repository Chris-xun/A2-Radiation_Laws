# Chris 05.03.2024
# calculate temperature from resistance data & the standard values given by the lab script


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.interpolate import CubicSpline
import ast
# import scienceplots
# plt.style.use('science')

# import data
given_data = np.loadtxt('data\\temp_resistance_relation_lab_script.csv', delimiter=',', skiprows=1)
taken_data = np.loadtxt('data\\resistance_to_temp.csv', delimiter=',', skiprows=1)  


# curve fitting the standard values, to allow for interpolation
def linear_func(x, m, c):
    return m * x + c
optimal_params, covariance = opt.curve_fit(linear_func, given_data[:, 1], given_data[:, 0], p0=[1, -3])
plt.plot(given_data[:, 1], linear_func(given_data[:, 1], *optimal_params), label='linear fit')
print('m:', optimal_params[0], 'c:', optimal_params[1])
def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c
optimal_params, covariance = opt.curve_fit(quadratic_func, given_data[:, 1], given_data[:, 0], p0=[1, 1, -3])
plt.plot(given_data[:, 1], quadratic_func(given_data[:, 1], *optimal_params), label='quadratic fit')
print('a:', optimal_params[0], 'b:', optimal_params[1], 'c:', optimal_params[2]) 

''' The fit parameters are found to be, where the quadratic fits almost perfectly:
m: 169.79246951960988 c: 237.8926282212558
a: -1.5992799065020011 b: 196.7283569841743 c: 139.4149679140698
'''

# plotting
plt.plot(given_data[:, 1], given_data[:, 0], 'x', color='blue', label='standard values')
plt.xlabel('Resistance [Ohm]')
plt.ylabel('Temperature [°K]')
plt.title('given Temperature - Resistance relation')
plt.grid()
plt.legend()
# plt.show()
plt.savefig('graphs\\temp_resistance_relation_lab_script.png')
plt.close()


# function to calculate temperature from resistance (T_R)
# the R_293 value is taken to be 1.2 +- 0.03
def cal_temp_from_normalised_resistance(resistance):
    return quadratic_func(resistance, -1.5992799065020011, 196.7283569841743, 139.4149679140698)

measured_normalised_resistance = np.array(taken_data[:, -1])
measured_temp_R = cal_temp_from_normalised_resistance(measured_normalised_resistance)
print(measured_temp_R)
'''[ 329.12395684  340.34849955  346.15007541  350.01619337  355.81297139
  371.25697245  390.53318673  413.62242288  448.16991599  494.07203272
  558.78465342  636.32868216  675.83812918  771.20238778  873.11381589
  950.28554163  981.36130856 1050.49053808 1077.65124908 1077.65124908
 1120.95869409 1231.98352432 1355.85640446 1472.95246785 1581.74791124
 1687.57700091 1797.20664741 1890.57313234 1996.04943957 2061.30683115]'''
 
# getting uncert
measured_temp_R_max = cal_temp_from_normalised_resistance(measured_normalised_resistance * 1.2 / (1.2+0.03))
measured_temp_R_min = cal_temp_from_normalised_resistance(measured_normalised_resistance * 1.2 / (1.2-0.03))
uncert = abs(measured_temp_R_max - measured_temp_R_min) / 2
print(uncert)
'''[ 4.70784553  4.98398332  5.12660185  5.2216008   5.36397914  5.74295013
  6.21522272  6.77983614  7.62243281  8.73782457 10.30217428 12.16397629
 13.10715786 15.36834216 17.76019691 19.55408608 20.27215365 21.86051249
 22.48113115 22.48113115 23.46662701 25.96980117 28.72198637 31.28272999
 33.62502094 35.86808304 38.15358753 40.06838615 42.19523825 43.49121418]'''

plt.errorbar(measured_normalised_resistance, measured_temp_R, yerr=uncert, fmt='x', color='red', label='measured values')
plt.xlabel('Resistance [Ohm]')
plt.ylabel('Temperature [°K]')
plt.title('Temperature - Resistance relation\n interpolated from standard values')
plt.grid()
plt.legend()
# plt.show()
plt.savefig('graphs\\temp_resistance_relation_interpolated.png')
plt.close()


# caluclating T_B from T_R
T_correction_data = np.loadtxt('data\\T_R_to_T_B_correction.csv', delimiter=',')
plt.plot(T_correction_data[:, 0], T_correction_data[:, 1], 'x', color='blue', label='standard values')
plt.title('T_R to T_B correction')
plt.xlabel('T_R Temperature [°K]')
plt.ylabel('T_B Temperature [°K]')
optimal_params, covariance = opt.curve_fit(linear_func, T_correction_data[:, 0], T_correction_data[:, 1])
plt.plot(T_correction_data[:, 0], linear_func(T_correction_data[:, 0], *optimal_params), label='linear fit')
plt.legend()
# plt.show()
plt.grid()
plt.savefig('graphs\\T_R_to_T_B_correction.png')
plt.close()
print('m:', optimal_params[0], 'c:', optimal_params[1])
'''m: 0.9558987321709527 c: 127.60388173355331'''

# calculating T_B from T_R
def cal_temp_B_from_temp_R(temp_R):
    return linear_func(temp_R, 0.9558987321709527, 127.60388173355331)



# plotting temperatures against current  
print(taken_data[:, 4])
cs = CubicSpline(taken_data[:, 4], cal_temp_B_from_temp_R(measured_temp_R))
plt.plot(taken_data[:, 4], cs(taken_data[:, 4]), label='cubic spline fit')
plt.plot(taken_data[:, 4], measured_temp_R, 'x', color='blue', label='T_R')
plt.plot(taken_data[:, 4], cal_temp_B_from_temp_R(measured_temp_R), 'x', color='red', label='T_B')
plt.xlabel('Current [A]')
plt.ylabel('Temperature [°K]')
plt.title('Temperature - Current relation')

# the values were determined later
arrays = []
with open('coil_cs.txt', 'r') as file:
    for line in file:
        # Use ast.literal_eval to safely evaluate the line as a Python list
        data_list = ast.literal_eval(line.strip())
        # Convert the list to a numpy array and append to our list of arrays
        arrays.append(np.array(data_list))
x = arrays[0]
y = arrays[1]
plt.plot(x, y, 'x', label='fitting bb spectrum, center', color = 'orange') 
cs = CubicSpline(x, y)
plt.plot(x, cs(x), label='cubic spline fit, bb spectrum', color = 'orange')

plt.grid()
plt.legend()
# plt.show()
plt.savefig('graphs\\temp_current_relation.png')
plt.close()


#will was here