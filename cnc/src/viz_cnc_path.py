import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cnc_path_viz(coordinates):
    # Extract X, Y, Z for plotting
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]

    # Plotting the 3D movement
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, label="CNC path")
    ax.scatter(x, y, z, c="r", marker="o")  # To highlight the points
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title("3D Visualization of CNC Path")
    ax.legend()
    plt.show()
    return fig, ax


# cnc_path_viz(df[['x', 'y', 'z']].to_numpy())
