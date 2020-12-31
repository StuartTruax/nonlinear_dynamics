import numpy as np
import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import ipdb

COLORMAP = 'winter'


def bifurcation_diagram(dynamics,variable_range,parameter_range, imperfection_parameter=0,
               equilibrium_threshold = 0.001):
    """
    Solve for the bifurcation diagram using a fixed grid search scheme for 1-D systems

    Arguments:
    ---------
    dynamics:  a function that has a variable as a first argument
               and a parameter as a second argument. Dynamics are of the form

               x_dot = dynamics(x,r)

    variable_range: an np array describing the range of the variable to evaluated
                   [x_min, x_max]

    parameter_range: an np array describing the range of the parameter to evaluated
                   [r_min, r_max]

    """

    equilibria_p = []
    equilibria_x = []

    for idx, p_val in enumerate(parameter_range):
        for jdx, x_val in enumerate(variable_range):

            x_dot = dynamics(x_val,p_val,imperfection_parameter)

            if np.abs(x_dot) <= equilibrium_threshold:
                equilibria_p.append(p_val)
                equilibria_x.append(x_val)


    return equilibria_p, equilibria_x


def bifurcation_diagram_adaptive(dynamics,variable_range,parameter_range, imperfection_parameter=0,
    equilibrium_threshold = 0.00001):
    """
    Solve for the bifurcation diagram using adaptive grid search for 1-D systems.
    Characterization of equilibiria also supported.

    Arguments:
    ---------
    dynamics:  a function that has a variable as a first argument
               and a parameter as a second argument. Dynamics are of the form

               x_dot = dynamics(x,r)

    variable_range: an np array describing the range of the variable to evaluated
                   [x_min, x_max]

    parameter_range: an np array describing the range of the parameter to evaluated
                   [r_min, r_max]

    """

    equilibria_p = []
    equilibria_x = []
    equilibria_type = []


    for idx, p_val in enumerate(parameter_range):

        #initialize variables
        x_cur = variable_range[1]
        x_prev = variable_range[0]

        delta_x = x_cur-x_prev

        x_dot = dynamics(x_cur,p_val)

        x_dot_grad = x_dot/delta_x

        x_dot_slice = []

        #adaptive grid loop
        while x_cur < variable_range[-1]:

            #variable_updates
            x_dot_grad_prev = x_dot_grad
            delta_x_prev = delta_x

            #obstain current point in state space
            x_dot = dynamics(x_cur,p_val,imperfection_parameter)

            #calculate quantities for x_dot_gradient criterion
            x_dot_grad = x_dot/delta_x
            x_dot_grad_ratio = np.abs(x_dot_grad/x_dot_grad_prev)

            #collect the (x,x_dot) ordered pair
            x_dot_slice.append([x_cur,x_dot])

            #update delta_x and x_cur
            delta_x = delta_x*x_dot_grad_ratio
            x_cur += delta_x

        ### post-hoc determination of equalibrium points
        x_dot_slice = np.array(x_dot_slice)

        #determine location and type of equilibrium by looking at sign change of x_dot
        x_dot_grad_sign = np.sign(x_dot_slice[:,1])

        ##mask for equilibrium points

        #stable points go from positive to negative
        stable_points_mask = ((np.roll(x_dot_grad_sign, 1) - x_dot_grad_sign) > 0)
        #unstable points go from negative to positive
        unstable_points_mask = ((np.roll(x_dot_grad_sign, 1) - x_dot_grad_sign) < 0)

        #first points are set to True as artifact of np.roll. Set to false to correct
        stable_points_mask[0]  = False
        unstable_points_mask[0] = False

        for stable_eq in x_dot_slice[stable_points_mask,:]:
            equilibria_p.append(p_val)
            equilibria_x.append(stable_eq[0])
            equilibria_type.append("stable")

        for unstable_eq in x_dot_slice[unstable_points_mask,:]:
            equilibria_p.append(p_val)
            equilibria_x.append(unstable_eq[0])
            equilibria_type.append("unstable")


        """
        plt.figure(figsize=(5,5))
        plt.plot(x_dot_slice[:,0], x_dot_slice[:,1])
        plt.ylabel(r"$x$", fontsize=20)
        plt.xlabel(r"$r$", fontsize=20)
        plt.grid()
        plt.show()
        """

    return equilibria_p, equilibria_x, equilibria_type


def phase_diagram_imperfection(dynamics,variable_range,parameter_range, imperfection_parameter_range):
        """
        Produce a phase diagram for several values of the imperfection parameter
        Iteratively solves for the equilibria at each ordered pair of the
        bifurcation and imperfection parameter to form a 3-surface for plotting.

        Arguments:
        ---------
        dynamics:  a function that has a variable as a first argument
                   and a parameter as a second argument. Dynamics are of the form

                   x_dot = dynamics(x,r)

        variable_range: an np array describing the range of the variable to evaluated
                       [x_min, x_max]

        parameter_range: an np array describing the range of the parameter to evaluated
                       [r_min, r_max]

        imperfection_parameter_range:  an np array describing the range of the imperfection
                       parameter to evaluated
                       [h_min, h_max]

        """

    colormap = plt.get_cmap(COLORMAP)
    colors = colormap(np.linspace(0,1.0,len(parameter_range)))

    #plot the 1-D dyanmics for several values of the bifurcation parameter
    plt.figure(figsize=(5,5))
    for idx, p_val in enumerate(parameter_range):
        #initialize variables
        x_cur = variable_range[1]
        x_prev = variable_range[0]

        delta_x = x_cur-x_prev

        x_dot = dynamics(x_cur,p_val)

        x_dot_grad = x_dot/delta_x

        x_dot_slice = []


        while x_cur < variable_range[-1]:

            #variable_updates
            x_dot_grad_prev = x_dot_grad
            delta_x_prev = delta_x

            #obstain current point in state space
            x_dot = dynamics(x_cur,p_val)

            #calculate quantities for x_dot_gradient criterion
            x_dot_grad = x_dot/delta_x
            x_dot_grad_ratio = np.abs(x_dot_grad/x_dot_grad_prev)

            #collect the (x,x_dot) ordered pair
            x_dot_slice.append([x_cur,x_dot])

            #update delta_x and x_cur
            delta_x = delta_x*x_dot_grad_ratio
            x_cur += delta_x


        x_dot_slice = np.array(x_dot_slice)
        plt.plot(x_dot_slice[:,0], x_dot_slice[:,1], color=colors[idx], label ="r=%0.3f"%p_val)

    #plot the imperfection parameter lines
    for h in imperfection_parameter_range:
        plt.plot(variable_range,h*np.ones(len(variable_range)), "-k",label ="h=%0.3f"%h)

    plt.ylabel(r"$\dot{x}$", fontsize=20)
    plt.xlabel(r"$x$", fontsize=20)
    plt.xlim([-2.5,2.5])
    plt.ylim([-1,4])
    plt.legend()
    plt.grid()
    plt.show()


def stability_diagram(dynamics,variable_range,parameter_range, imperfection_parameter_range):
    """
    Produce a stability diagram for a 1-D system.
    Iteratively solves for the equilibria at each ordered pair of the
    bifurcation and imperfection parameter to form a 3-surface for plotting.

    Arguments:
    ---------
    dynamics:  a function that has a variable as a first argument
               and a parameter as a second argument. Dynamics are of the form

               x_dot = dynamics(x,r)

    variable_range: an np array describing the range of the variable to evaluated
                   [x_min, x_max]

    parameter_range: an np array describing the range of the parameter to evaluated
                   [r_min, r_max]

    imperfection_parameter_range:  an np array describing the range of the imperfection
                   parameter to evaluated
                   [h_min, h_max]

    """

    #vectors
    P_stable = np.array([])
    H_stable = np.array([])
    X_stable = np.array([])

    P_unstable = np.array([])
    H_unstable = np.array([])
    X_unstable = np.array([])

    #iterate through the imperfection parameter values
    for h_index,h in enumerate(imperfection_parameter_range):

        #find the equilibria
        equilibria_p, equilibria_x, equilibria_type = \
        bifurcation_diagram_adaptive(dynamics,variable_range,parameter_range,h)

        #convert to np.arrays for masking
        equilibria_p = np.array(equilibria_p)
        equilibria_x = np.array(equilibria_x)
        equilibria_type = np.array(equilibria_type)

        #separate stable from unstable equilibria
        stable_indices =  np.where(equilibria_type == "stable")
        unstable_indices = np.where(equilibria_type == "unstable")


        #collect coordinates, each in a long list
        P_stable = np.concatenate((P_stable,equilibria_p[stable_indices]))
        H_stable = np.concatenate((H_stable,[h for _ in range(len(equilibria_p[stable_indices]))]))
        X_stable = np.concatenate((X_stable, equilibria_x[stable_indices]))

        P_unstable = np.concatenate((P_unstable,equilibria_p[unstable_indices]))
        H_unstable = np.concatenate((H_unstable,[h for _ in range(len(equilibria_p[unstable_indices]))]))
        X_unstable = np.concatenate((X_unstable, equilibria_x[unstable_indices]))



    #plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    colormap = plt.get_cmap(COLORMAP)
    colors = colormap(np.linspace(0.1,0.9,2))

    ax.scatter(P_stable, H_stable, X_stable, s=0.1, color = colors[0])
    ax.scatter(P_unstable, H_unstable, X_unstable, s=0.1, color = colors[-1])
    ax.set_xlabel('r')
    ax.set_ylabel('h')
    ax.set_zlabel('x')

    plt.show()








def saddle_node(x,r):
    return r+x**2

def supercritical_pitchfork(x,r, h=0):
    return h+r*x-x**3

def subcritical_pitchfork(x,r, h=0):
    return h+r*x+x**3


def plot_bifurcation(equilibria_p, equilibria_x,equilibria_type):
    """
    Function for plotting equilibrium points as a function of the bifurcation
    parameter.
    """

    colormap = plt.get_cmap(COLORMAP)
    colors = colormap(np.linspace(0.1,0.9,2))

    plt.figure(figsize=(5,5))

    for i in range(len(equilibria_p)):
        if equilibria_type[i] == "stable":
            plt.plot(equilibria_p[i], equilibria_x[i], '.', color=colors[0], alpha=1)
        else:
            plt.plot(equilibria_p[i], equilibria_x[i], '.', color=colors[-1], alpha=1)


    plt.legend((colors[0], colors[-1]),
           ('Stable Equilibria', 'Unstable Equilibria'),
           loc='lower left',
           ncol=3,
           fontsize=8)


    legend_elements = [Line2D([0], [0], lw=4, label='Stable Equilibria',
                        color=colors[0]),
                   Line2D([0], [0], lw=4, label='Unstable Equilibria',
                          color=colors[-1])]

    plt.legend(handles=legend_elements, loc='lower left')
    plt.grid()
    plt.ylabel(r"$x$", fontsize=20)
    plt.xlabel(r"$r$", fontsize=20)
    plt.show()


if __name__=="__main__":


    parameter_range = np.linspace(-1.0,1.0,1000)
    variable_range = np.linspace(-10.0,10.0,100)


    equilibria_p, equilibria_x,equilibria_type  = bifurcation_diagram_adaptive(supercritical_pitchfork, \
                                               variable_range,parameter_range)
    """
    plot_bifurcation(equilibria_p, equilibria_x,equilibria_type)


    equilibria_p, equilibria_x,equilibria_type  = bifurcation_diagram_adaptive(subcritical_pitchfork, \
                                               variable_range,parameter_range)

    plot_bifurcation(equilibria_p, equilibria_x,equilibria_type)


    equilibria_p, equilibria_x,equilibria_type  = bifurcation_diagram_adaptive(saddle_node, \
                                               variable_range,parameter_range)

    plot_bifurcation(equilibria_p, equilibria_x,equilibria_type)
    """

    imperfection_parameter_range = np.linspace(-1,2,200)
    parameter_range = np.linspace(-1.0,1.0,200)


    """
    phase_diagram_imperfection(supercritical_pitchfork, variable_range,parameter_range, \
                         imperfection_parameter_range)
    """
    stability_diagram(subcritical_pitchfork,variable_range,parameter_range, imperfection_parameter_range)
