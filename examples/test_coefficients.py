import numpy as np
import dedalus.public as d3
from mpi4py import MPI

# -----------------------------
# Domain and bases
# -----------------------------
Nx, Ny = 256, 256
Lx, Ly = 2*np.pi, 2*np.pi

coords = d3.CartesianCoordinates('x', 'y')
dist   = d3.Distributor(coords, dtype=np.float64, comm=MPI.COMM_WORLD)

# Simulation bases: RealFourier 
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=1.5)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=1.5)

u = dist.VectorField(coords, bases=(xbasis, ybasis), name='u')  # 2 components (ux, uy)

# -----------------------------
# Construct a synthetic 2D velocity field from sinusoids
# -----------------------------
# Use a streamfunction ψ so the field is exactly divergence-free:
#   ψ(x,y) = sin(kx*x) * sin(ky*y)
#   ux =  ∂ψ/∂y =  ky * cos(ky*y) * sin(kx*x)
#   uy = -∂ψ/∂x = -kx * cos(kx*x) * sin(ky*y)

kx_mode = 2
ky_mode = 3

x, y = dist.local_grids(xbasis, ybasis)

psi = np.sin(kx_mode * x) * np.sin(ky_mode * y)
ux  =  ky_mode * np.cos(ky_mode * y) * np.sin(kx_mode * x)
uy  = -kx_mode * np.cos(kx_mode * x) * np.sin(ky_mode * y)

u['g'][0] = ux
u['g'][1] = uy

# Theoretical KE
KE_theoretical = 0.5 * (2**2 + 3**2)*np.pi**2
if dist.comm_cart.rank == 0:
    print(f"Theoretical KE     : {KE_theoretical: .8e}")

# -----------------------------
# 1) Total kinetic energy in physical space
# -----------------------------
u.require_grid_space()

# Local grid spacings (periodic, no endpoint)
dx = x[1, 0] - x[0, 0]
dy = y[0, 1] - y[0, 0]

ux_g = u['g'][0]
uy_g = u['g'][1]

KE_local_grid = 0.5 * np.sum(ux_g**2 + uy_g**2) * dx * dy
KE_grid = dist.comm_cart.allreduce(KE_local_grid, op=MPI.SUM)

if dist.comm_cart.rank == 0:
    print(f"Total KE (grid)     : {KE_grid: .8e}")

# --------------------
# 2) Total KE from RealFourier coefficients (Parseval)
# --------------------
# Move to coefficient space
u.require_coeff_space()

ux_c = u['c'][0]
uy_c = u['c'][1]

# RealFourier mode numbers for each coefficient entry
kx = dist.local_modes(xbasis)   # shape broadcastable to ux_c
ky = dist.local_modes(ybasis)   # same for y

# RealFourier Parseval weights:
# 1D: (1/N) sum f^2 = a(0)^2 + 1/2 * sum_{k>0}(a(k)^2 + b(k)^2)
# so: weight = 1 if k==0, else 1/2
w_x = np.where(kx == 0, 1.0, 0.5)
w_y = np.where(ky == 0, 1.0, 0.5)
weight = w_x * w_y  # 2D weight factor

coeff_sq = ux_c**2 + uy_c**2

# --- Total KE from RealFourier coeffs (check against grid & theory) ---
local_ke_spec = 0.5 * np.sum(coeff_sq * weight) * Lx * Ly
KE_spec = dist.comm_cart.allreduce(local_ke_spec, op=MPI.SUM)

if dist.comm_cart.rank == 0:
    print(f"Total KE (spectral) : {KE_spec: .8e}")


# --- 1D radial spectrum from RealFourier coeffs ---

# Use integer radial wavenumber index:
# k_index = round( sqrt(kx^2 + ky^2) )
Nx_c, Ny_c = ux_c.shape[-2:]

# Integer mode numbers corresponding to RealFourier pairs:
nx_index = np.arange(Nx_c) // 2   # 0,0,1,1,2,2,...
ny_index = np.arange(Ny_c) // 2   # 0,0,1,1,2,2,...

NX, NY = np.meshgrid(nx_index, ny_index, indexing='ij')
k_mag = np.sqrt(NX**2 + NY**2)
k_index = np.rint(k_mag).astype(int)

# Per-coefficient KE *per unit area* (drop Lx*Ly if you want that normalization)
e_coeff = 0.5 * coeff_sq * weight * Lx * Ly

# Flatten for binning
k_flat = k_index.ravel()
e_flat = e_coeff.ravel()

kmax = int(k_flat.max())
E_local = np.bincount(k_flat, weights=e_flat, minlength=kmax+1)

# Sum across MPI ranks
E_shell = np.empty_like(E_local)
dist.comm_cart.Allreduce(E_local, E_shell, op=MPI.SUM)

# Convert to physical k = |k| * 2π/L
# Here Lx == Ly, so:
dk = 2*np.pi / Lx
k_phys = dk * np.arange(len(E_shell))

if dist.comm_cart.rank == 0:
    print("k (index)    E_shell(k)")
    for k_idx, Ek in enumerate(E_shell[:10]):  # first 10 shells as a sanity check
        print(f"{k_idx:3d}       {Ek: .8e}")
