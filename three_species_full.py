import numpy as np
from dedalus import public as de
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import glob
import subprocess


def euclidean_distance(vec1, vec2):
    """
    Compute the Euclidean distance between two numeric vectors.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.linalg.norm(vec1 - vec2)


def cultural_distance(cultures_dict, name1, name2):
    """
    Takes a dictionary of cultures and two culture names (keys), and returns the
    Euclidean distance between their mu vectors.
    """
    mu1 = cultures_dict[name1]['mu']
    mu2 = cultures_dict[name2]['mu']
    return euclidean_distance(mu1, mu2)


def sign(f):
    # Safe division with epsilon
    return f / (de.Abs(f) + 1e-6)

def max_field_change(field, field_old):
    field.change_scales(1)
    field_old.change_scales(1)
    return np.max(np.abs(field['g'] - field_old['g']))


# def spectral_filter(field, alpha=18.0):
#     """
#     Exponential spectral filter for Dedalus 3 (1D Fourier).
#     `alpha` controls filter strength.
#     """
#     # Make sure we're in coefficient space
#     field.require_coeff_space()

#     # Get mode numbers from basis
#     basis = field.bases[0]
#     N = basis.N
#     Lx = basis.domain.lengths[0]
#     k = np.fft.fftfreq(N, d=Lx/N)

#     # Construct damping mask
#     k_max = np.max(np.abs(k))
#     sigma = np.exp(-alpha * (np.abs(k) / k_max)**8)

#     # Apply damping to spectral data
#     field.data[:] *= sigma[:, np.newaxis]


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu)/sigma)**2)


def get_sim_time(path):
    with h5py.File(path, 'r') as f:
        # Adjust if you saved sim_time as a field instead
        return f['tasks']['sim_time'][0, 0]

### ------- Parameters -------
Lx = 1000.0 # Length of the domain (km)
Nx = 2**10 # Number of grid points
dealias = 3/2 # Dealiasing factor

stop_sim_time = 3.0 # Stop time for the simulation (years)
timestep = 0.001  # Time step size (years)
snapshot_interval = 10 # How often to take snapshots (in iterations)
dt_max = 0.1
dt_min = 1e-5
safety = 0.5
threshold = 0.05  # define a reasonable change in n_i

min_density = 1e-6  # Minimum allowed density to avoid numerical issues
epsilon = 1e-6 # smooth positive epsilon to avoid log(0) or division by zero

# Settings
frame_rate = 5
input_pattern = "/home/chaddyfynn/scripts/social-dynamics/frames/frame_%04d.png"

# Output names
palette_path = "palette.png"
gif_path = "output.gif"

## Cultural parameters (vector differences assumed precomputed as scalars)
# These live in a vector space where
# mu in R^8 (8 values with no foced anti-correlations)
# [equality, markets, globe, nation, liberty, authority, progress, tradition]
cultures = {
    "Test Culture": {
        "mu": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # Cultural vector (8 values test) (dimensionless)
        "lambda": 1.0, # Contentiousness parameter (dimensionless)
        "eta": 1.0, # Cultural diffusion control (dimensionless) (maybe get rid - can be controlled by D) [Set to 1.0 for now]
        "D": 1.0, # Diffusion coefficient (km^2 year^-1)
        "kappa": 1.0 # Ideological response frequency (year^-1)
    },
    "Anarchism": {
        "mu": [0.9, 0.1, 0.8, 0.2, 1.0, 0.0, 0.9, 0.1],
        "lambda": 0.7,
        "eta": 1.0,
        "D": 0.2,
        "kappa": 0.9
    },
    "Liberalism": {
        "mu": [0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1],
        "lambda": 0.4,
        "eta": 1.0,
        "D": 2.0,
        "kappa": 0.8
    },
    "Fascism": {
        "mu": [0.1, 0.9, 0.2, 0.8, 0.2, 0.9, 0.1, 0.9],
        "lambda": 0.95,
        "eta": 1.0,
        "D": 20.0,
        "kappa": 1.0
    },
    "Religious Traditionalism": {
        "mu": [0.3, 0.5, 0.3, 0.7, 0.4, 0.6, 0.2, 1.0],
        "lambda": 0.6,
        "eta": 1.0,
        "D": 0.30,
        "kappa": 0.4
    },
    "Eco-Socialism": {
        "mu": [0.9, 0.1, 0.9, 0.1, 0.7, 0.3, 0.95, 0.2],
        "lambda": 0.5,
        "eta": 1.0,
        "D": 0.30,
        "kappa": 0.7
    },
    "Technocracy": {
        "mu": [0.5, 0.6, 0.7, 0.3, 0.4, 0.6, 0.8, 0.2],
        "lambda": 0.3,
        "eta": 1.0,
        "D": 1.0,
        "kappa": 0.6
    }
}



field_names = ["Eco-Socialism", "Fascism", "Liberalism"]

mu_12 = cultural_distance(cultures, field_names[0], field_names[1])
mu_13 = cultural_distance(cultures, field_names[0], field_names[2])
mu_23 = cultural_distance(cultures, field_names[1], field_names[2])

eta_1 = cultures[field_names[0]]['eta']
eta_2 = cultures[field_names[1]]['eta']
eta_3 = cultures[field_names[2]]['eta']

lambda_1 = cultures[field_names[0]]['lambda']
lambda_2 = cultures[field_names[1]]['lambda']
lambda_3 = cultures[field_names[2]]['lambda']

D1 = cultures[field_names[0]]['D']
D2 = cultures[field_names[1]]['D']
D3 = cultures[field_names[2]]['D']

kappa_1 = cultures[field_names[0]]['kappa']
kappa_2 = cultures[field_names[1]]['kappa']
kappa_3 = cultures[field_names[2]]['kappa']

C = 1.0

# Basis, Coordinates, and Distributor Setup
coord = de.Coordinate('x')
x = coord.coords[0]  # x coordinate
dist = de.Distributor(coord, dtype=np.float64)
# x_basis = de.Chebyshev(coord, size=Nx, bounds=(0, Lx), dealias=dealias)
x_basis = de.Fourier(coord, size=Nx, bounds=(0, Lx), dealias=dealias, dtype=np.float64)

# Fields
n1 = dist.Field(name='n1', bases=x_basis)
n2 = dist.Field(name='n2', bases=x_basis)
n3 = dist.Field(name='n3', bases=x_basis)

n1_old = dist.Field(name='n1_old', bases=x_basis)
n2_old = dist.Field(name='n2_old', bases=x_basis)
n3_old = dist.Field(name='n3_old', bases=x_basis)

# Auxiliary fields for second derivatives
# n1_xx = dist.Field(name='n1_xx', bases=x_basis)
# n2_xx = dist.Field(name='n2_xx', bases=x_basis)
# n3_xx = dist.Field(name='n3_xx', bases=x_basis)

dx = lambda A: de.Differentiate(A, coord)
epsilon = 1e-6
Abs = lambda f: (f*f + epsilon)**0.5  # <== This replaces de.Abs!
dt = lambda A: de.TimeDerivative(A, coord)

# ------- Problem Definition -------
# Diffusion terms and potential-derived reaction terms
# dV_ij/dn_i = 2*lambda_i*mu_ij*n_i(x)*n_j(x)**2/sqrt(C**2 + n_ij(x)**2) - lambda_i*mu_ij*n_i(x)**2*n_ij(x)*n_j(x)**2/(C**2 + n_ij(x)**2)**(3/2)
# e.g.     "+ 2*lambda_1*mu_12*n1*n2**2/(C**2 + Abs(n1 - n2)**2)**0.5 - lambda_1*mu_12*n1**2*Abs(n1 - n2)*n2**2/(C**2 + Abs(n1 - n2)**2)**(3/2)"

# In terms of raw derivatives, we can write:
# eq_n1 = (
#     "dt(n1)"
#     " = "
#     " + D1*eta_1*dx(dx(n1))"
#     "+ 2*lambda_1*mu_12*n1*n2**2/(C**2 + Abs(n1 - n2)**2)**0.5"
#     "- lambda_1*mu_12*n1**2*Abs(n1 - n2)*n2**2/(C**2 + Abs(n1 - n2)**2)**(3/2)"
#     "+ 2*lambda_1*mu_13*n1*n3**2/(C**2 + Abs(n1 - n3)**2)**0.5"
#     "- lambda_1*mu_13*n1**2*Abs(n1 - n3)*n3**2/(C**2 + Abs(n1 - n3)**2)**(3/2)"
# )

# eq_n2 = (
#     "dt(n2)"
#     " = "
#     " + D2*eta_2*dx(dx(n2))"
#     "+ 2*lambda_2*mu_12*n2*n1**2/(C**2 + Abs(n2 - n1)**2)**0.5"
#     "- lambda_2*mu_12*n2**2*Abs(n2 - n1)*n1**2/(C**2 + Abs(n2 - n1)**2)**(3/2)"
#     "+ 2*lambda_2*mu_23*n2*n3**2/(C**2 + Abs(n2 - n3)**2)**0.5"
#     "- lambda_2*mu_23*n2**2*Abs(n2 - n3)*n3**2/(C**2 + Abs(n2 - n3)**2)**(3/2)"
# )

# eq_n3 = (
#     "dt(n3)"
#     " = "
#     " + D3*eta_3*dx(dx(n3))"
#     "+ 2*lambda_3*mu_13*n3*n1**2/(C**2 + Abs(n3 - n1)**2)**0.5"
#     "- lambda_3*mu_13*n3**2*Abs(n3 - n1)*n1**2/(C**2 + Abs(n3 - n1)**2)**(3/2)"
#     "+ 2*lambda_3*mu_23*n3*n2**2/(C**2 + Abs(n3 - n2)**2)**0.5"
#     "- lambda_3*mu_23*n3**2*Abs(n3 - n2)*n2**2/(C**2 + Abs(n3 - n2)**2)**(3/2)"
# )

eq_n1 = (
    "dt(n1) = "
    "+ D1*eta_1*dx(dx(n1))"
    "+ kappa_1*( "
    " + 2*lambda_1*mu_12*n1*n2**2/(C**2 + Abs(n1 - n2)**2)**0.5"
    " - lambda_1*mu_12*n1**2*Abs(n1 - n2)*n2**2/(C**2 + Abs(n1 - n2)**2)**(3/2)"
    " + 2*lambda_1*mu_13*n1*n3**2/(C**2 + Abs(n1 - n3)**2)**0.5"
    " - lambda_1*mu_13*n1**2*Abs(n1 - n3)*n3**2/(C**2 + Abs(n1 - n3)**2)**(3/2)"
    ")"
)

eq_n2 = (
    "dt(n2) = "
    "+ D2*eta_2*dx(dx(n2))"
    "+ kappa_2*( "
    " + 2*lambda_2*mu_12*n2*n1**2/(C**2 + Abs(n2 - n1)**2)**0.5"
    " - lambda_2*mu_12*n2**2*Abs(n2 - n1)*n1**2/(C**2 + Abs(n2 - n1)**2)**(3/2)"
    " + 2*lambda_2*mu_23*n2*n3**2/(C**2 + Abs(n2 - n3)**2)**0.5"
    " - lambda_2*mu_23*n2**2*Abs(n2 - n3)*n3**2/(C**2 + Abs(n2 - n3)**2)**(3/2)"
    ")"
)

eq_n3 = (
    "dt(n3) = "
    "+ D3*eta_3*dx(dx(n3))"
    "+ kappa_3*( "
    " + 2*lambda_3*mu_13*n3*n1**2/(C**2 + Abs(n3 - n1)**2)**0.5"
    " - lambda_3*mu_13*n3**2*Abs(n3 - n1)*n1**2/(C**2 + Abs(n3 - n1)**2)**(3/2)"
    " + 2*lambda_3*mu_23*n3*n2**2/(C**2 + Abs(n3 - n2)**2)**0.5"
    " - lambda_3*mu_23*n3**2*Abs(n3 - n2)*n2**2/(C**2 + Abs(n3 - n2)**2)**(3/2)"
    ")"
)




equations = [eq_n1, eq_n2, eq_n3]


namespace = {
    'n1': n1,
    'n2': n2,
    'n3': n3,
    'dx': dx,
    'dt': dt,
    'Abs': Abs,
    'eta_1': eta_1,
    'eta_2': eta_2,
    'eta_3': eta_3,
    'lambda_1': lambda_1,
    'lambda_2': lambda_2,
    'lambda_3': lambda_3,
    'mu_12': mu_12,
    'mu_13': mu_13,
    'mu_23': mu_23,
    'C': C,
}


problem = de.IVP([n1, n2, n3], namespace=locals())
# problem = de.IVP([n1, n2, n3, n1_xx, n2_xx, n3_xx], namespace=locals())
for eqn in equations:
    problem.add_equation(eqn)

# Auxilliary equations for second derivatives
# problem.add_equation("n1_xx = dx(dx(n1))")
# problem.add_equation("n2_xx = dx(dx(n2))")
# problem.add_equation("n3_xx = dx(dx(n3))")


# ------- Boundary Conditions -------
# For simplicity, we can use Neumann BCs (zero gradient at boundaries)
# problem.add_equation("dx(n1)(x='left') = 0")
# problem.add_equation("dx(n1)(x='right') = 0")
# problem.add_equation("dx(n2)(x='left') = 0")
# problem.add_equation("dx(n2)(x='right') = 0")
# problem.add_equation("dx(n3)(x='left') = 0")
# problem.add_equation("dx(n3)(x='right') = 0")

# Alternatively, we can set the values at the boundaries to zero
# problem.add_equation("n1(x='left') = 0")
# problem.add_equation("n1(x='right') = 0")
# problem.add_equation("n2(x='left') = 0")
# problem.add_equation("n2(x='right') = 0")
# problem.add_equation("n3(x='left') = 0")
# problem.add_equation("n3(x='right') = 0")

# Or we can set the second derivatives to zero (Neumann BCs)
# problem.add_equation("n1_xx(x='left') = 0")
# problem.add_equation("n1_xx(x='right') = 0")
# problem.add_equation("n2_xx(x='left') = 0")
# problem.add_equation("n2_xx(x='right') = 0")
# problem.add_equation("n3_xx(x='left') = 0")
# problem.add_equation("n3_xx(x='right') = 0")

# ------- Solver Setup -------
print("Number of variables:", len(problem.variables))
print("Number of equations:", len(problem.equations))
solver = problem.build_solver(de.RK443)
solver.stop_sim_time = stop_sim_time
solver.ok = True

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=timestep, max_writes=100)
snapshots.add_task(n1)
snapshots.add_task(n2)
snapshots.add_task(n3)

x_grid = dist.local_grid(x_basis)  # get grid in physical space
x_field = dist.Field(name='x', bases=x_basis)
x_field['g'] = x_grid

snapshots.add_task(x_field, name='x')

# sim_time = dist.Field(name='sim_time')
# sim_time['g'] = solver.sim_time
# snapshots.add_task(sim_time, name='sim_time')

sim_time_field = dist.Field(name='sim_time')
snapshots.add_task(sim_time_field, name='sim_time')

# ------- Initial Conditions -------
x = x_basis.local_grid(dist, 1)  # shape (Nx,)

sigma = Lx / 10  # width of each gaussian, ~1/10 domain
centers = [Lx * 0.5 - Lx * 0.2, Lx * 0.5, Lx * 0.5 + Lx * 0.2]  # left, middle, right

n1['g'] = gaussian(x, centers[0], sigma)  # peak near left end
n2['g'] = 0.1 * gaussian(x, centers[1], sigma)  # peak in the middle
n3['g'] = gaussian(x, centers[2], sigma)  # peak near right end

plt.figure(figsize=(10, 6))
plt.plot(x, n1['g'], label=f'{field_names[0]} (n1)', color='tab:blue')
plt.plot(x, n2['g'], label=f'{field_names[1]} (n2)', color='tab:red')
plt.plot(x, n3['g'], label=f'{field_names[2]} (n3)', color='tab:green')

plt.xlabel('x')
plt.ylabel('Initial Concentration')
plt.title('Sanity Check: Initial Conditions')
plt.legend()
plt.grid(True)
plt.show()


# ------- Time-stepping Loop -------
solver.step(0.0)
while solver.proceed and solver.ok:
    sim_time_field['g'] = solver.sim_time

    # Backup current state
    n1.change_scales(1)
    n2.change_scales(1)
    n3.change_scales(1)
    n1_old['g'] = n1['g'].copy()
    n2_old['g'] = n2['g'].copy()
    n3_old['g'] = n3['g'].copy()
    n1.change_scales(dealias)
    n2.change_scales(dealias)
    n3.change_scales(dealias)


    solver.step(timestep)

    # Escape if simulation becomes unstable
    if (
        np.isnan(n1['g']).any() or
        np.isnan(n2['g']).any() or
        np.isnan(n3['g']).any()
    ):
        print(f"⚠️ NaN detected at sim time t = {solver.sim_time:.5f}, stopping simulation.")
        solver.ok = False

    if any(not np.all(np.isfinite(n['g'])) for n in [n1, n2, n3]):
        print("💀 Non-finite value detected, aborting.")
        solver.ok = False

    # Adaptive timestep control
    n1.change_scales(1)
    n2.change_scales(1)
    n3.change_scales(1)
    max_d1 = np.max(np.abs(n1['g'] - n1_old['g']))
    max_d2 = np.max(np.abs(n2['g'] - n2_old['g']))
    max_d3 = np.max(np.abs(n3['g'] - n3_old['g']))
    n1.change_scales(dealias)
    n2.change_scales(dealias)
    n3.change_scales(dealias)
    max_change = max(max_d1, max_d2, max_d3)

    # Update timestep
    if max_change > threshold:
        timestep *= safety * (threshold / max_change)
        timestep = max(timestep, dt_min)
    elif max_change < threshold / 10:
        timestep *= 1.1
        timestep = min(timestep, dt_max)

    # Spectral filtering (optional)
    # for n in [n1, n2, n3]:
    #     n.require_coeff_space()
    #     spectral_filter(n)

    # hard clamp
    # for n in [n1, n2, n3]:
    #     n['g'] = np.maximum(n['g'], min_density)

    # soft clamp
    # for n in [n1, n2, n3]:
    #     n['g'] = np.log1p(np.exp(n['g'])) + epsilon  # smooth approx of max(0, n)

    if solver.iteration % snapshot_interval == 0:
        print(f"Iter {solver.iteration:05d}, t = {solver.sim_time:.3f}")
    if solver.iteration % 10 == 0:
        print(f"max(n1) = {np.nanmax(n1['g'])}, min(n1) = {np.nanmin(n1['g'])}")


# ------- Plot Final State -------
n1_final = solver.state[0]['g'].copy()
n2_final = solver.state[1]['g'].copy()
n3_final = solver.state[2]['g'].copy()

n1_final = n1['g'].copy()
n2_final = n2['g'].copy()
n3_final = n3['g'].copy()

x = x_basis.local_grid(dist, dealias)

plt.figure(figsize=(10, 6))
plt.plot(x, n1_final, label=field_names[0], color='tab:blue')
plt.plot(x, n2_final, label=field_names[1], color='tab:red')
plt.plot(x, n3_final, label=field_names[2], color='tab:green')
plt.xlabel('x')
plt.ylabel('n(x)')
plt.title('Final Cultural Distributions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

snapshot_files = sorted(glob.glob('snapshots/snapshots_*.h5'), key=get_sim_time)

for i, snap_file in enumerate(snapshot_files):
    with h5py.File(snap_file, 'r') as f:
        # Load spatial grid (should now be saved properly)
        x = f['tasks']['x'][0]  # shape: (Nx,) or maybe (1, Nx)

        # Load field data for n1
        n1 = f['tasks']['n1'][0]  # first time slice
        n2 = f['tasks']['n2'][0]
        n3 = f['tasks']['n3'][0]

        time_val = f['tasks']['sim_time'][:][0,0]  # or slice accordingly

        plt.clf()
        plt.plot(x, n1, label=f'n1 ({field_names[0]})')
        plt.plot(x, n2, label=f'n2 ({field_names[1]})')
        plt.plot(x, n3, label=f'n3 ({field_names[2]})')
        plt.legend()
        sim_time_data = f['tasks']['sim_time'][:]
        # Usually 1 entry per snapshot: shape = (1, 1, 1, 1) for 0D field
        sim_time_val = sim_time_data[0, 0]
        plt.title(f"Time: {sim_time_val:.3f}")
        plt.xlabel('x')
        plt.ylabel('nᵢ(x)')
        plt.savefig(f'frames/frame_{i:04d}.png')


# Step 1: Generate palette (color optimization)
subprocess.run([
    "ffmpeg",
    "-framerate", str(frame_rate),
    "-i", input_pattern,
    "-vf", "palettegen",
    "-y",  # overwrite if exists
    palette_path
], check=True)

# Step 2: Create the GIF using the palette
subprocess.run([
    "ffmpeg",
    "-framerate", str(frame_rate),
    "-i", input_pattern,
    "-i", palette_path,
    "-lavfi", "paletteuse",
    "-an",  # no audio
    "-y",
    gif_path
], check=True)

print(f"GIF saved to {gif_path}")