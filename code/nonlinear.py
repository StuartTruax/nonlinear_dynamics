import numpy as np
import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.font_manager as fm
from scipy import integrate
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import ipdb

COLORMAP = 'winter'

## helpful reference: https://scipy-cookbook.readthedocs.io/items/LoktaVolterraTutorial.html

def lotka_volterra(x,t,params):
    return np.array([params[0]*x[0]-params[1]*x[0]*x[1],
                     -params[2]*x[1]+params[1]*params[3]*x[0]*x[1]])

def strogatz_predator_prey(x,t,params):
    return np.array([x[0]*(params[0]-x[0]-params[1]*x[1]),
                     x[1]*(params[2]-x[0]-x[1])])



def conservative_system(x,t,params=None):
    return np.array([x[1],
                    x[0]-x[0]**3])

def conservative_potential(x):
    return -.5*(x**2)+0.25*(x**4)


def jacobian(dynamics,point, epsilon_ball_radius):
    pass


def trajectory(dynamics,time_points, x_0, params):

    X, info = integrate.odeint(dynamics, x_0, time_points, args=(params,), full_output=True)
    return X


def energy_of_conservative_system(x, x_dot, potential, mass):
    return potential(x)+0.5*mass*(x_dot**2)


def energy_surface(trajectories, x_range, y_range, mass, potential):
    """
    Builds an energy surface from a collection of trajectories.
    """

    level_trajectories = []

    #for each trajectory
    #get the energy
    for trajectory in trajectories:
        level_surface = np.empty((trajectory.shape[0],trajectory.shape[1]+1))
        level_surface[:,0:2]= trajectory
        level_surface[:,2] = potential(trajectory[:,0])+0.5*mass*(trajectory[:,1]**2)
        level_trajectories.append(level_surface)


    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(x_range, y_range)
    Es = np.array(energy_of_conservative_system(np.ravel(X), np.ravel(Y), potential, mass))
    E = Es.reshape(X.shape)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, E, cmap=cm.winter,
                       linewidth=0, antialiased=False, alpha=0.2)

    colormap = plt.get_cmap(COLORMAP)
    colors = colormap(np.linspace(0,1.0,len(trajectories)))

    #sort the trajectories by energy level
    level_trajectories.sort(key= lambda x: x[0][2])

    for i,level_trajectory  in enumerate(level_trajectories):
        plt.plot(level_trajectory [:,0],level_trajectory[:,1], level_trajectory[:,2],color = colors[i])


    ax.set_xlabel('$x$', fontsize=20, rotation=150)
    ax.set_ylabel('$\dot{x}$', fontsize=20)
    ax.set_zlabel(r'$E$', fontsize=20, rotation=60)

    #ax.set_zlim(0, 4)
    #ax.set_xlim(0, 4)
    #ax.set_ylim(0, 4)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    #fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def generate_phase_space(dynamics, params, x_range,y_range, equilibrium_threshold = 0.01):
    """
    Finds the time derivatives of the phase space variables at every
    point in a fixed grid of the 2-D space.

    Arguments:
    ---------

    dynamics: function which defines the dynamics of the system

    parameters: vector of parameters to the function

    x_range: np array of x values defining the fixed grid of the phase space

    y_range: np array of y values defining the fixed grid of the phase space

    Returns:
    -------

    X: an np matrix with the values of the x coordinates of the phase space

    Y: aan np matrix with the values of the y coordinates of the phase space

    U: an np matrix with the magnitudes of the time change of the x-component

    V: an np matrix with the magnitudes of the time change of the y-component
    """

    #X varies column-wise
    #Y varies row-wise
    X, Y = np.meshgrid(x_range,y_range)

    U = np.empty(X.shape)
    V = np.empty(X.shape)

    #fixed points
    fixed_points = np.empty((1,2))

    #iterate through the variable range, evaluating the system matrix
    # at each point
    for i,x in enumerate(x_range):
        for j,y in enumerate(y_range):

            vector = dynamics([x,y],0,params)

            U[j,i] = vector[0]
            V[j,i] = vector[1]


            if np.hypot(vector[0], vector[1]) < equilibrium_threshold:
                fixed_points = np.vstack((fixed_points,vector))



    # used to color the phase diagram by vector magnitude
    M  = (np.hypot(U, V))

    return X, Y, U, V, M, fixed_points


def phase_diagram(X, Y, U,V, M, fixed_points, trajectories = None):
    """
    Plot the phase diagram for a 2-D system

    Arguments:
    ---------

    X: an np matrix with the values of the x coordinates of the phase space

    Y: aan np matrix with the values of the y coordinates of the phase space

    U: an np matrix with the magnitudes of the time change of the x-component

    V: an np matrix with the magnitudes of the time change of the y-component

    trajectories:  an array of np arrays of ordered pairs for trajectories to
                   be plotted over the vector field

    """

    plt.figure(figsize=(5,5))
    plt.quiver(X, Y, U,V, M, pivot='mid',cmap=cm.jet)

    colormap = plt.get_cmap(COLORMAP)
    colors = colormap(np.linspace(0,1.0,len(trajectories)))

    for i,trajectory in enumerate(trajectories):
        plt.plot(trajectory[:,0],trajectory[:,1],color = colors[i])


    for fixed_point in fixed_points:
        plt.plot(fixed_point[0],fixed_point[1], 'k.',markersize=10)


    plt.xlabel(r"$x$", fontsize=20)
    plt.ylabel(r"$y$", fontsize=20)
    plt.show()




if __name__=="__main__":



    ###### Lotka-Volterra system
    params = [1., 0.1, 1.5, 0.75]
    t = np.linspace(0, 15,  1000)
    x_0 = np.array([10, 5])
    delta_x = np.array([10, 4])

    x_range = np.linspace(0,50.0,40)
    y_range = np.linspace(0,50.0,40)

    initial_points = [x_0+delta_x*i for i in range(5)]

    trajectories = []

    for i in range(len(initial_points)):
        trajectories.append(trajectory(lotka_volterra,t,initial_points[i],params))


    X, Y, U,V, M, fixed_points = generate_phase_space(lotka_volterra, params, x_range,y_range)

    phase_diagram(X, Y, U,V, M, fixed_points, trajectories)

    #####Strogatz's predator prey system
    x_range = np.linspace(0,4,20)
    y_range = np.linspace(0,4,20)
    t = np.linspace(0, 40,  1000)
    x_0 = np.array([0.1, 0.1])

    N_trajectories = 10
    x_0_x_range = np.linspace(0,4,N_trajectories )
    x_0_y_range = np.linspace(0,4,N_trajectories )

    initial_points = []
    for i in range(N_trajectories):
        for j in range(N_trajectories):
            initial_points+=[np.array([x_0_x_range[i],x_0_y_range[j]])]

    params = [3.,2., 2.]

    trajectories = []

    for i in range(len(initial_points)):
        trajectories.append(trajectory(strogatz_predator_prey,t,initial_points[i],params))


    X, Y, U,V, M, fixed_points = generate_phase_space(strogatz_predator_prey, params, x_range,y_range)

    phase_diagram(X, Y, U,V, M, fixed_points, trajectories)


    ###### Conservative system
    N_trajectories = 10
    x_range = np.linspace(-2,2,50)
    y_range = np.linspace(-2,2,50)
    t = np.linspace(0, 40,  1000)
    x_0 = np.array([0, 0])

    x_0_x_range = np.linspace(-1,1,N_trajectories )
    x_0_y_range = np.linspace(-1,1,N_trajectories )


    initial_points = []
    for i in range(N_trajectories):
        for j in range(N_trajectories):
            initial_points+=[np.array([x_0_x_range[i],x_0_y_range[j]])]


    trajectories = []

    for i in range(len(initial_points)):
        trajectories.append(trajectory(conservative_system,t,initial_points[i],None))


    X, Y, U,V, M, fixed_points = generate_phase_space(conservative_system, None, x_range,y_range)

    phase_diagram(X, Y, U,V, M, fixed_points, trajectories)

    #plot the energy surface for the conservative system
    energy_surface(trajectories, x_range, y_range, 1.0, conservative_potential)
