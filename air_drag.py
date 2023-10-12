import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

def acceleration(xdot, zdot, c_D):
    return np.array([
        -1/(2 * m) * air_density * c_D * cross_sect_area * xdot * np.sqrt(xdot**2 + zdot**2),
        -g - 1/(2 * m) * air_density * c_D * cross_sect_area * zdot * np.sqrt(xdot**2 + zdot**2)
    ])

# Constants and initial conditions
air_density = 1.225
m = 4400
g = 9.81
initial_hort_speed = 146.6
initial_vert_pos = 9632
final_height = 572
cross_sect_area = 0.398
drag_coeffs = [0.04, 1.15]
dt = 0.01  # time step

# Initialize variables for each drag coefficient
results = []

# Runge-Kutta 4th order method
for c_D in drag_coeffs:
    time = [0]
    x = [0]
    z = [initial_vert_pos]
    xdot = [initial_hort_speed]
    zdot = [0]
    
    while z[-1] > final_height:
        i = len(time) - 1
        k1 = dt * acceleration(xdot[i], zdot[i], c_D)
        k2 = dt * acceleration(xdot[i] + 0.5 * k1[0], zdot[i] + 0.5 * k1[1], c_D)
        k3 = dt * acceleration(xdot[i] + 0.5 * k2[0], zdot[i] + 0.5 * k2[1], c_D)
        k4 = dt * acceleration(xdot[i] + k3[0], zdot[i] + k3[1], c_D)
        
        new_xdot = xdot[-1] + 1/6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        new_zdot = zdot[-1] + 1/6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        
        new_x = x[-1] + new_xdot * dt
        new_z = z[-1] + new_zdot * dt
        
        time.append(time[-1] + dt)
        x.append(new_x)
        z.append(new_z)
        xdot.append(new_xdot)
        zdot.append(new_zdot)
        
    results.append((time, x, z))

# Plotting z(t)
plt.figure(1)
plt.title('Projectile Height vs Time')
for i, (time, x, z) in enumerate(results):
    plt.plot(time, z, label=f'c_D = {drag_coeffs[i]}')
    
    # Find the time when z reaches z_f
    time_at_zf = next((t for t, height in zip(time, z) if height <= final_height), None)
    
    if time_at_zf is not None:
        plt.annotate(f'Time at z_f for c_D = {drag_coeffs[i]}: {time_at_zf:.2f} s',
                     xy=(time_at_zf, final_height),
                     xytext=(time_at_zf - 40, final_height + 1000*(i+1)),  # Adjusted position
                     arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.legend()
plt.show()

xy_loc = [[-40,-1000],[-30,-3200]]

# Plotting x(t)
plt.figure(2)
plt.title('Projectile Horizontal Distance vs Time')
for i, (time, x, z) in enumerate(results):
    plt.plot(time, x, label=f'c_D = {drag_coeffs[i]}')
    
    # Get the final distance
    final_distance = x[-1]
    
    plt.annotate(f'Final distance for c_D = {drag_coeffs[i]}: {final_distance:.2f} m',
                 xy=(time[-1], final_distance),
                 xytext=(time[-1] + xy_loc[i][0], final_distance + xy_loc[i][1]),  # Adjusted position
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.xlabel('Time (s)')
plt.ylabel('Horizontal Distance (m)')
plt.legend()
plt.show()

# -- Enola Gay Escape --

v = 157
R = 1450
dz = 9060

# Case 1 - Actual Plane Ride
# v_plane = 157 m/s -> R = 1450 m, v_s = 360 m/s, \Delta z = 9060 m
# Given: \Delta t_{AC} = 43.16s, x_n = 6262.81m

# Case 2 - Theoretical Minimum for the following assumptions:
# v_plane = 157 m/s -> R = 1450 m, max(v_s) = 417 m/s, \Delta z = 9060 m
# \Delta t_{AC} = 48.01s, x_n = 5321.11m
cases = [
    {'v_s': 360, 'dt_AC': 43.16, 'x_n': 6262.81, 'label': 'Case 1 - Actual Plane Ride'},
    {'v_s': 417, 'dt_AC': 48.01, 'x_n': 5321.11, 'label': 'Case 2 - Theoretical Minimum'}
]

for case in cases:
    v_s = case['v_s']
    dt_AC = case['dt_AC']
    x_n = case['x_n']
    label = case['label']

    def f(s, theta):
        return s - np.sqrt((x_n - R*np.sin(theta) - v*(dt_AC - R*theta/v + s/v_s)*np.cos(theta)) ** 2 + 
                           (R*(1-np.cos(theta)) + v*(dt_AC - R*theta/v + s/v_s) * np.sin(theta)) ** 2 + 
                           dz ** 2)

    # Initialize variables
    theta_values = np.linspace(0, np.pi, 100)
    s_values = []

    # Solve for s at different theta values
    for theta in theta_values:
        s_initial_guess = 0  # Initial guess for s
        s_solution, = fsolve(f, s_initial_guess, args=(theta,))
        s_values.append(s_solution)

    # Find maximal value of s and corresponding theta
    max_s = max(s_values)
    theta_at_max_s = theta_values[s_values.index(max_s)]

    # Plotting s(theta)
    plt.figure()
    plt.plot(np.degrees(theta_values), s_values, label='s(theta)')
    plt.axhline(y=14000, color='r', linestyle='--', label='Safe boundary')
    plt.annotate(f'Maximal value: s = {max_s:.2f} m,\nTheta = {np.degrees(theta_at_max_s):.2f} degrees',
                 xy=(np.degrees(theta_at_max_s), max_s),
                 xytext=(90, 15000),  # Adjusted position
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.xlabel('Theta (degrees)')
    plt.ylabel('s (m)')
    plt.title(f'Maximal value of s with respect to Theta ({label})')
    plt.legend()
    plt.show()
