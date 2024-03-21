import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

def interpolate_heatmap(data, x_dist, y_dist):
    # padding the data with zeros
    data = np.pad(data, ((1, 1), (1, 1)), mode='constant')
    
    # Get dimensions of the input data
    n, m = data.shape
    
    # Define grid points for interpolation
    x = np.arange(0, m * x_dist, x_dist)
    y = np.arange(0, n * y_dist, y_dist)
    
    # Create interpolating function
    f = interp2d(x, y, data, kind="cubic")
    
    # Generate finer grid for smoother heatmap
    x_new = np.arange(0, (m - 1) * x_dist, 0.1)
    y_new = np.arange(0, (n - 1) * y_dist, 0.1)
    
    # Interpolate data on finer grid
    interpolated_data = f(x_new, y_new)
    
    # Create heatmap
    plt.imshow(interpolated_data, cmap='hot', interpolation='nearest', origin='lower',
               extent=[0, (m - 1) * x_dist, 0, (n - 1) * y_dist])
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Coil Emmisivity')
    plt.savefig('coil_emmisivity.png')
    plt.show()


# # Generate random data
# N = 5  # Number of rows
# M = 5  # Number of columns
# x_distance = 5  # Distance between points in X direction (pixels)
# y_distance = 5  # Distance between points in Y direction (pixels)
# data = np.random.rand(N, M)
# # Add ghost points at the edges
# data_with_ghost = np.pad(data, ((1, 1), (1, 1)), mode='constant')


# possible example data
def example():
    N = 3
    M = 6
    x_distance = 5
    y_distance = 5
    data = np.array([[0.2, 0.5, 0.2],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.6, 0.2],
                    [0.3, 0.7, 0.3],
                    [0.2, 0.5, 0.3],
                    [0.1, 0.4, 0.2]])                              

    # Interpolate and plot heatmap
    interpolate_heatmap(data, x_distance, y_distance)

