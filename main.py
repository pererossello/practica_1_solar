import jax
import jax.numpy as jnp

DATAPATH = '/home/prossello/Documents/ULL/subjects/q3/fisica_solar/code/entregable_1/data/Convection3D_20km_30s_z0_420km.h5'

G_SURF = 274 * 1e-3

GAMMA = 5/3
R = 8.314
MU = 1.247e-3

M = 241-1
N = 288

DX = 2e4 * 1e-3
DT = 30

L = DX*N
print(L)
T = DT*M
print(T)
print(T/3600)


def get_cs(temperature):
    return jnp.sqrt(R*GAMMA*temperature/MU) * 1e-3

fourier_norm = "forward"

@jax.jit
def get_k_1D():
    k_1D = jnp.fft.fftfreq(N) * N * 2 * jnp.pi / L
    return k_1D

@jax.jit
def get_k():
    k_1D = get_k_1D()
    k = jnp.linalg.norm(jnp.array(jnp.meshgrid(*[k_1D] * 2, indexing="ij")), axis=0)
    return k

@jax.jit
def spatial_fft(f):
    f_k = L**2 * jnp.fft.fftn(f, norm=fourier_norm, axes=(1, 2))
    return f_k

@jax.jit
def temporal_fft(f):
    f_k = T * jnp.fft.rfft(f, norm=fourier_norm, axis=0)
    return f_k

@jax.jit
def get_omega(positive_only=True):
    omega_1D = jnp.fft.fftfreq(M) * M * 2 * jnp.pi / T
    if positive_only:
        omega_1D = jnp.abs(omega_1D[:M//2+1])
    return omega_1D

def get_power(vz):
    vz_hat = spatial_fft(temporal_fft(vz))
    return jnp.abs(vz_hat)**2 / L**6 

def get_k_omega_power(vz, n_k_bins=100):

    k = get_k()
    omega = get_omega()

    pow_spec = get_power(vz)
    pow_spec_2D = jnp.zeros((pow_spec.shape[0], n_k_bins-1))

    min_k = 0 
    max_k = jnp.max(k)

    edges = jnp.linspace(min_k, max_k, n_k_bins)
    k_bins =  edges[:-1] + 0.5 * jnp.diff(edges)

    counts, _ = jnp.histogram(k, bins=edges)
    
    for i in range(M):
        nonorm_hist, _ = jnp.histogram(k, weights=pow_spec[i,...], bins=edges)
        radial_distro = nonorm_hist / counts
        pow_spec_2D = pow_spec_2D.at[i,:].set(radial_distro)

    return k_bins, omega, pow_spec_2D
