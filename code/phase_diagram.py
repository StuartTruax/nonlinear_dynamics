import numpy as np
import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import ipdb

COLORMAP = 'winter'


def solve_eigen(A):
    """
    Solves the eigenvalues and eignevectors of A.

    Arguments:
    ---------
    A: np matrix of linear system

    Returns:
    -------
    lambdas: np array of eigenvalues

    eigenvectors: np matrix of eigenvectors
    """

    lambdas = []
    eigenvectors = []


    lambdas, eigenvectors = np.linalg.eig(A)

    return lambdas, eigenvectors

def generate_phase_space(A, x_range,y_range):
    """
    Finds the time derivatives of the phase space variables at every
    point in a fixed grid of the 2-D space.

    Arguments:
    ---------

    A: np linear system matrix to be valuated

    x_range: np array of x values defining the fixed grid of the phase space

    y_range: np array of y values defining the fixed grid of the phase space

    Returns:
    -------

    X: an np matrix with the values of the x coordinates of the phase space

    Y: aan np matrix with the values of the y coordinates of the phase space

    U: an np matrix with the magnitudes of the time change of the x-component

    V: an np matrix with the magnitudes of the time change of the y-component

    fixed_points:  an np array of ordered pairs for the equiblibria of the system.
                   Fixed points plotted as solid dots.


    """

    #X varies column-wise
    #Y varies row-wise
    X, Y = np.meshgrid(x_range,y_range)

    U = np.empty(X.shape)
    V = np.empty(X.shape)

    #iterate through the variable range, evaluating the system matrix
    # at each point
    for i,x in enumerate(x_range):
        for j,y in enumerate(y_range):

            vector = np.matmul(A,[x,y])

            U[j,i] = vector[0]
            V[j,i] = vector[1]

    #find fixed points
    try:
        fixed_points = np.linalg.solve(A,np.zeros((2,1)))
        fixed_points = fixed_points.flatten()
    except:
        #matrix is singular
        fixed_points = np.array([])

    print("System Fixed Points:")
    print(fixed_points)


    return X, Y, U, V, fixed_points



def phase_diagram(X, Y, U,V, fixed_points):
    """
    Plot the phase diagram for a 2-D system

    Arguments:
    ---------

    X: an np matrix with the values of the x coordinates of the phase space

    Y: aan np matrix with the values of the y coordinates of the phase space

    U: an np matrix with the magnitudes of the time change of the x-component

    V: an np matrix with the magnitudes of the time change of the y-component

    fixed_points:  an np array of ordered pairs for the equiblibria of the system.
                   Fixed points plotted as solid dots.

    """

    plt.figure(figsize=(5,5))
    plt.quiver(X, Y, U,V, scale=6, units="xy")

    if len(fixed_points) > 0:
        plt.plot(fixed_points[0],fixed_points[1], 'k.',markersize=20)

    plt.xlabel(r"$x$", fontsize=20)
    plt.ylabel(r"$y$", fontsize=20)
    plt.show()



if __name__=="__main__":

    ## general examples
    A_ls = np.array([[-2,0],[0,-1]])
    A_eq = np.array([[-1,0],[0,-1]])
    A_ls_zero = np.array([[-0.5,0],[0,-1]])
    A_zero = np.array([[0,0],[0,-1]])
    A_geq_zero = np.array([[1,0],[0,-1]])

    A_degenerate = np.array([[-1,0],[0,-1]])

    A_center = np.array([[0,1],[-1,0]])
    A_spiral = np.array([[1,1],[-1,1]])

    ## harmonic oscillator parameters and examples
    k = 1
    m = 1
    zeta = 5e-1

    A_damped_harmonic_osc = np.array([[0,1],[-(k/m),-2*zeta*np.sqrt(k/m)]])
    A_undamped_harmonic_osc = np.array([[0,1],[-(k/m),0]])

    x_range = np.linspace(-1.0,1.0,10)
    y_range = np.linspace(-1.0,1.0,10)

    ## Romeo and Juliet examples
    A_mutual_indifference = np.array([[-1.5,1],[1,-1.5]])
    A_explosive_relationship = np.array([[-1,1.5],[1.5,-1]])
    A_love_hate = np.array([[0,1],[-1,0]])
    A_stale_romeo = np.array([[0,0],[-1,1]])
    A_opposites_attract = np.array([[-1,1],[-1,1]])
    A_same_styles = np.array([[-1,1],[1,-1]])
    A_out_of_touch_mutual = np.array([[0,1],[1,0]])
    A_out_of_touch_osc = np.array([[0,1],[-1,0]])


    #one-line assigment for above examples
    A = A_out_of_touch_osc

    print("System Eigen-properties: ")
    print(solve_eigen(A))

    X, Y, U,V, fixed_points = generate_phase_space(A , x_range,y_range)

    phase_diagram(X, Y, U,V, fixed_points)
