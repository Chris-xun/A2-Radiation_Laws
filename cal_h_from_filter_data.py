# Chris 08.03.2024
# calculating the planck constant from the filter data obtained

# importing 
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import functions as f
from scipy.interpolate import CubicSpline

lamdas =  550e-9  #[598.91e-9]
c = 2.99792e8
h = 6.626e-34
k_b = 1.380649e-23

def linear(x, a, b):
    result = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] == 0:
            result[i] = np.nan
        else:
            result[i] = a*x[i] + b
    return a*x + b

# importing the data
data = np.loadtxt('data\\photodetector_data.csv', delimiter=',', skiprows=1)

lamp_current = data[:, 0]
lamp_current_uncert = data[:, 1]
filter1 = data[:, 4] * 1e3
# filter2 = data[:, 6] * 1e3
filter_uncert = data[:, 5] * 1e3


# cupic spline interpolation to get the T_B from the current measured
taken_data = np.loadtxt('data\\resistance_to_temp.csv', delimiter=',', skiprows=1) 
cs = CubicSpline(taken_data[:, 4], f.cal_temp_B_from_temp_R(f.cal_temp_from_normalised_resistance(taken_data[:, -1])))


# removing the 0 values from the data
mask = filter1 != 0
filter1 = filter1[mask]
filter_uncert = filter_uncert[mask]
reduced_lamp_current = lamp_current[mask]
reduced_lamp_current_uncert = lamp_current_uncert[mask]

# calculating
filter1 = np.log(filter1)
filter_uncert = filter_uncert/filter1
temperture = cs(reduced_lamp_current)
x_values = 1 / temperture
x_values_uncert = 1 / temperture**2 * reduced_lamp_current_uncert
optimal_params, cov_matrix = opt.curve_fit(linear, x_values, filter1, sigma=filter_uncert)
h_measured = -1 * k_b * optimal_params[0] * lamdas / c
print('h measured:', h_measured, 'fractional error:' , abs(h - h_measured) / h )

# plotting the data
plt.title('Planck constant from filter data\n Ln(Signal) vs 1/T')
plt.errorbar(x_values, filter1, yerr=abs(filter_uncert), xerr=x_values_uncert, fmt='o', label='measured values (1/T)')
plt.plot(x_values, linear(x_values,*optimal_params), label='y = {:.2f}x + {:.2f}'.format(optimal_params[0], optimal_params[1]))
plt.xlabel('Lamp Current (A)')
plt.ylabel('ln(Signal)')
plt.grid()
plt.legend()
plt.show()