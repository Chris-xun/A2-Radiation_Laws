# Chris 19.03.2024
# analysising the sections of the lamp, rather than the whole lamp as one

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import functions as f


# skipping tbe irrelevant lines
def load_file_custom(path):
    with open(path, 'r') as file:
        # Skip the first 48 lines
        for _ in range(48):
            next(file)
        # Read the rest into a list, except the last line
        lines = file.readlines()[:-1]

    # Convert lines to a list of lists of floats
    data = [list(map(float, line.split(','))) for line in lines]
    
    # Convert to NumPy array
    np_data = np.array(data)

    return np_data

# blackbody radiation function
def blackbody_radiation(lambdas, T , A):
    c = 2.99792e8
    h = 6.626e-34
    k_b = 1.380649e-23
    value =  A * (2*h*c**2) / ((lambdas)**5 * (np.exp(h*c / ((lambdas)*k_b*T)) - 1))
    return value



# all the files names
current_numbers = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
colors = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'pink', 'yellow', 'brown', 'grey']

# current_numbers = [0.4]
# colors = ['red']

temp = []
# plotting all the files onto 1 graph for comparison
for current, color in zip(current_numbers, colors):
    # importing the data
    data = load_file_custom(f'data\section_of_coil__current_vs_temp\half_way_{current}A.csv')
    
    # normalising the data
    x = data[:, 0] * 1e-9
    y = data[:, 1] / np.trapz(data[:,1], x)

    data_slicing_posiion = len(x) // 5 * 3
    optimal_params, cov_matrix = opt.curve_fit(blackbody_radiation, x[:data_slicing_posiion], y[:data_slicing_posiion], p0=[1000, 0.5])
    print(optimal_params[0])
    temp = np.append(temp, optimal_params[0])
    # print(blackbody_radiation(x, optimal_params[0]))
    fit_y = blackbody_radiation(x, *optimal_params) #/ np.max(blackbody_radiation(x, *optimal_params))
    plt.plot(x, y, label=str(current)+'A', color=color, alpha = 0.5)
    plt.plot(x, fit_y, color=color)

plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.ylim(0, 1e7)
plt.legend()
plt.show()

plt.plot(current_numbers, temp)
plt.xlabel('Current (A)')
plt.ylabel('Temperature (K)')
plt.show()