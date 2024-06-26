# Chris 08.03.2024
# calculating the planck constant from the filter data obtained

# importing 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import scipy.optimize as opt
import functions as f
from scipy.interpolate import CubicSpline
import ast

lambdas = [598.9e-9, 551.0e-9, 500.6e-9, 456.5e-9, 398.4e-9]
c = 2.99792e8
h = 6.626e-34
k_b = 1.380649e-23

def round_to_significant_figure(x, n):
    if x == 0:
        return 0
    else:
        # Determine the order of magnitude of the number
        order_of_magnitude = np.floor(np.log10(np.abs(x)))
        # Round the number to the first significant figure
        rounded_number = x * 10**(n-order_of_magnitude)
        rounded_number = np.round(rounded_number)
        rounded_number = rounded_number / 10**(n-order_of_magnitude)
        return rounded_number

def reduce_by_first_nonzero(arr):
    # Convert to a numpy array if not already
    arr = np.array(arr)
    
    # Find the first non-zero value
    non_zero_values = arr[arr != 0]
    if len(non_zero_values) == 0:
        # If there are no non-zero values, return the original array
        return arr
    
    first_non_zero = non_zero_values[0] + 1e-10
    
    # Subtract the first non-zero value from every element
    reduced_arr = arr - first_non_zero
    
    return reduced_arr


# importing the data    # 'data\\gain_photodetector.csv' or 'data\\filter_measure\\all.csv'
data = np.loadtxt('data\\filter_measure\\all.csv', delimiter=',', skiprows=1)

# Create a ScalarFormatter object
formatter = ScalarFormatter(useMathText=True)  # useMathText=True to use math text for scientific notation
formatter.set_scientific(True)  # Enable scientific notation
formatter.set_powerlimits((-1,1))  # You can adjust these limits based on your data

# Apply the formatter to the y-axis
plt.gca().xaxis.set_major_formatter(formatter)

data_index = [2,3,4,5,6]
uncert_index = [1,1,1,1,1]
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i in range(len(data_index)):
    lambda1 = lambdas[i]
    lamp_current = data[:, 0]
    lamp_current_uncert = np.array([0.0005 for _ in range(len(lamp_current))]) # this is the actual uncert
    filter1 = data[:, data_index[i]] * 1e3
    filter1 = reduce_by_first_nonzero(filter1)
    filter_uncert = data[:, uncert_index[i]] * 1e3
    # plt.plot(lamp_current, np.array(filter1), 'x')
    # plt.show()


    # cupic spline interpolation to get the T_B from the current measured
    taken_data = np.loadtxt('data\\resistance_to_temp.csv', delimiter=',', skiprows=1) 
    y = f.cal_temp_B_from_temp_R(f.cal_temp_from_normalised_resistance(taken_data[:, -1]))
    x = taken_data[:, 4]
    # plt.plot(x,y)
    # plt.savefig('graphs\cubic_spline_current_to_TB.png')
    # plt.close()
    cs = CubicSpline(x,y)
    
    arrays = []
    with open('coil_cs.txt', 'r') as file:
        for line in file:
            # Use ast.literal_eval to safely evaluate the line as a Python list
            data_list = ast.literal_eval(line.strip())
            # Convert the list to a numpy array and append to our list of arrays
            arrays.append(np.array(data_list))
    x = arrays[0]
    y = arrays[1]
    cs_bb_spectrum = CubicSpline(x, y)

    cs = cs_bb_spectrum


    # removing the 0 values from the data
    mask = filter1 > 0
    filter1 = filter1[mask]
    filter_uncert = filter_uncert[mask]
    reduced_lamp_current = lamp_current[mask]
    reduced_lamp_current_uncert = lamp_current_uncert[mask]

    # calculating
    # print(filter1)
    signal = np.log(filter1)
    filter_uncert = filter_uncert/filter1
    temperture = cs(reduced_lamp_current)
    # plt.close()
    # plt.title('1/temmp thingy')
    # plt.plot(1/temperture)
    # plt.show()
    x_values = 1 / temperture
    x_values_uncert = 1 / temperture**2 * reduced_lamp_current_uncert
    optimal_params, cov_matrix = opt.curve_fit(f.linear_func, x_values, signal, sigma=x_values_uncert)
    h_measured = -1 * k_b * optimal_params[0] * lambda1 / c
    h_measured_uncert = 1 * k_b * np.sqrt(cov_matrix[0,0]) * lambda1 / c
    print('\nfor filter: ', lambda1,'h measured:', h_measured, 'fractional deviation:' , abs(h - h_measured) / h )

    # plotting the data
    plt.title('Planck constant from filter data\n Ln(Signal) vs 1/T')
    # plt.plot(x_values, signal,'x', label='(1/T) for filter: ' + str(round_to_significant_figure(lambda1*1e9, 4)) + "nm, h = (" + str(round_to_significant_figure(h_measured*1e34,3)) + " $\pm$ " + str(round_to_significant_figure(h_measured_uncert*1e34,1)) + ')$\\times 10^{-34}$')
    plt.errorbar(x_values, signal, yerr=abs(filter_uncert), xerr=x_values_uncert, fmt='o', color=colors[i], label='(1/T) for filter: ' + str(round_to_significant_figure(lambda1*1e9, 5)) + "nm, h = (" + str(round_to_significant_figure(h_measured*1e34,2)) + " $\pm$ " + str(round_to_significant_figure(h_measured_uncert*1e34,0)) + ')$\\times 10^{-34}$')
    # plt.errorbar(x_values, signal, yerr=abs(filter_uncert), xerr=x_values_uncert, fmt='o', label="(1/T) for filter: {:.3f}, h = {:.3f} $\pm$ {:.1f} ".format(lambda1, h_measured, h_measured_uncert))

    plt.plot(x_values, f.linear_func(x_values,*optimal_params), color=colors[i]) #, label='y = {:.2f}x + {:.2f}'.format(optimal_params[0], optimal_params[1]))
    plt.xlabel('Inverse Temperature (1/T) ($K^{-1}$)')
    plt.ylabel('ln(Signal)')
    
plt.grid()
plt.legend(loc=1)
plt.savefig('graphs\\planck_constant_from_voltage_filter_data_calculated_temp_current_relation.png')
plt.show()