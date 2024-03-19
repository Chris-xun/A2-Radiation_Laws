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



# all the files names
current_numbers = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

# plotting all the files onto 1 graph for comparison
for current in current_numbers:
    # importing the data
    data = load_file_custom(f'data\section_of_coil__current_vs_temp\half_way_{current}A.csv')
    x = data[:-2, 0]
    y = data[:-2, 1]
    plt.plot(x, y, label=str(current)+'A')

plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.legend()
plt.show()
