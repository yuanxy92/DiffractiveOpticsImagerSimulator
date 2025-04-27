import numpy as np

params = {}
params['structure_to_phase_coeffs'] = [6.051, -0.02033, 2.26, 1.371E-5, -0.002947, 0.797]
params['nanometers'] = 1E-9
params['whole_dim'] = 2285
params['padding_dim'] = 2500

num_pixels = params['whole_dim']  # Needed for 0.5 mm diameter aperture
pixelsX = num_pixels
pixelsY = num_pixels
params['pixelsX'] = pixelsX
params['pixelsY'] = pixelsY
params['wavelength_nominal'] = 452E-9
params['pitch'] = 350E-9
params['Lx'] = 1 * params['pitch']
params['Ly'] = params['Lx']
dx = params['Lx'] # grid resolution along x
dy = params['Ly'] # grid resolution along x
xa = np.linspace(0, pixelsX - 1, pixelsX) * dx # x axis array
xa = xa - np.mean(xa) # center x axis at zero
ya = np.linspace(0, pixelsY - 1, pixelsY) * dy # y axis vector
ya = ya - np.mean(ya) # center y axis at zero
[y_mesh, x_mesh] = np.meshgrid(ya, xa)
params['x_mesh'] = x_mesh
params['y_mesh'] = y_mesh
params['pixels_aperture'] = 6e-4 / params['pitch']

# Function to add noise to the phase map
def add_noise(duty, noise_level_random, noise_level_sys):
    """
    Adds Gaussian noise to the phase map.
    
    Parameters:
    - phase_map: 2D numpy array, the original phase map.
    - noise_level: float, the standard deviation of the noise relative to the range of the phase map.
    
    Returns:
    - noisy_phase_map: 2D numpy array, the phase map with added noise.
    """
    # noise = np.random.normal(0, noise_level_random * np.ptp(duty), duty.shape) + noise_level_sys * 2 * (np.random.random() - 0.5)
    noise = noise_level_random * 2 * (np.random.rand(duty.shape[0], duty.shape[1]) - 0.5) + noise_level_sys * 2 * (np.random.random() - 0.5)
    noisy_duty = duty + noise
    return noisy_duty

def phase_from_duty_and_lambda(duty, params):
    p = params['structure_to_phase_coeffs']
    lam = params['lam0'] / params['nanometers']
    phase = p[0] + p[1]*lam + p[2]*duty + p[3]*lam**2 + p[4]*lam*duty + p[5]*duty**2
    return phase * 2 * np.pi