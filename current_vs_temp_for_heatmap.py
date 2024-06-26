# Chris 19.03.2024
# analysising the sections of the lamp, rather than the whole lamp as one
# calculating temperature of each section of the coil
# the current is   ?

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import functions as f
import os
import coil_heatmap as h
import scienceplots
plt.style.use(['science', 'notebook'])
plt.rcParams['font.family'] = 'Times New Roman'

# defining the functions
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

folder = 'data\\coil_position\\coil_position_data'
files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

# storing the temp data
temp = np.zeros((6, 3))
for file in files:
    # importing the data
    data = load_file_custom(os.path.join(folder, file))
    
    # normalising the data
    x = data[:, 0] * 1e-9
    y = data[:, 1] / np.trapz(data[:,1], x)

    # data_slicing_posiion = len(x) // 5 * 3
    data_slicing_posiion = (np.where(y>0.1e7)[0][-1] - np.where(y>0.1e7)[0][0]) // 6 + np.where(y>0.1e7)[0][0]
    optimal_params, cov_matrix = opt.curve_fit(blackbody_radiation, x[:data_slicing_posiion], y[:data_slicing_posiion], p0=[1000, 0.5])
    print(optimal_params[0], np.sqrt(cov_matrix[0, 0]))
    
    
    # print(blackbody_radiation(x, optimal_params[0]))
    fit_y = blackbody_radiation(x, *optimal_params) #/ np.max(blackbody_radiation(x, *optimal_params))
    plt.plot(x, y, label=data, alpha = 0.5)
    plt.plot(x, fit_y, label='fit', linestyle='--') 
    plt.savefig('{}\\graphs\\{}.png'.format(folder, file))
    plt.close()
    
    
    #storing the temp
    # getting the index
    if 'L' in file:
        i = 0
    elif 'M' in file:
        i = 1
    elif 'R' in file:
        i = 2
    j = int(file[1])
    
    temp[j, i] = optimal_params[0]
print(temp)

# plotting the data in a heat map
# h.interpolate_heatmap(temp, 1, 1)

# trying to showcase the information more clearly 
plt.figure(figsize=(5,2))
plt.title('Tungsten Coil Heat Distrabution', fontsize = 25)
im = plt.imshow(temp, cmap='plasma', interpolation='gaussian', origin = 'lower')
for i in range(temp.shape[0]):
    for j in range(temp.shape[1]):
        plt.text(j, i, f'{int(round(temp[i, j]))}', ha='center', 
                 va='bottom', color='white', fontsize=20, fontweight='bold', )
y_ticks_mm = np.linspace(0, 2.5, 6)  # Generate ticks at 0, 0.5, 1, 1.5, 2, and 2.5 mm
y_ticks_units = np.linspace(0, 5, 6)  # Corresponding tick positions in original units

x_ticks_mm = np.linspace(0, 2, 3)  # Generate ticks at 0, 0.5, 1, 1.5, 2, and 2.5 mm
x_ticks_units = np.linspace(0, 2, 3)

plt.yticks(y_ticks_units, y_ticks_mm, fontsize = 20)  # Set the ticks with the rescaled values
plt.xticks(x_ticks_units, x_ticks_mm, fontsize = 20) 
cbar = plt.colorbar(im)
cbar.set_label('Temperature [K]', rotation=270, labelpad=25, fontsize = 25)
cbar.ax.tick_params(labelsize = 20)

plt.xlabel('X position [cm]', fontsize = 25)
plt.ylabel('Y position [cm]', fontsize = 25)
plt.show()

print(temp[:, 1])