from scipy.special import comb
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def bernstein_poly(i: int, n: int, t: np.ndarray) -> np.ndarray:
    """
    Calculate the Bernstein polynomial of n, i as a function of t.
    
    Args:
        i (int): The index of the Bernstein basis polynomial
        n (int): The degree of the polynomial
        t (np.ndarray): Parameter values at which to evaluate the polynomial
        
    Returns:
        np.ndarray: Values of the Bernstein polynomial at t
    """
    return comb(n, i) * (t**(n-i)) * (1 - t)**i

def bezier_curve(points: list, nTimes: int = 1000) -> tuple:
    """
    Generate a Bezier curve from control points.
    
    Args:
        points (list): List of control points as [x,y] coordinates
        nTimes (int): Number of points to generate along the curve
        
    Returns:
        tuple: (x-coordinates, y-coordinates) of points along the Bezier curve
    """
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)
    polynomial_array = np.array(
        [bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)]
    )

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def plot_curve_using_control_points(img: np.ndarray, control_points: list) -> plt.Figure:
    """
    Plot a Bezier curve on an image using control points.
    
    Args:
        img (np.ndarray): Input image
        control_points (list): List of control points for the Bezier curve
        
    Returns:
        plt.Figure: Matplotlib figure with the plotted curve
    """
    codes = [mpath.Path.MOVETO] + [mpath.Path.CURVE4]*(len(control_points)-1)
    path = mpath.Path(control_points, codes)
    patch = mpatches.PathPatch(path, facecolor='none', edgecolor='red', linewidth=2)
    
    ax = plt.gca()
    ax.imshow(img)
    ax.add_patch(patch)
    
    return plt.gcf() 