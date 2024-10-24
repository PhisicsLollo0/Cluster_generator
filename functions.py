import numpy as np

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def d_to_m(d):
    m = 5*np.log10(d) - 5
    return m

def m_to_d(m):
    d = 10**(1+m/5)
    return d

def mag_to_flux(magnitudes, band):
    if band == 'G':  ZP = 25.6874  #G
    if band == 'BP': ZP = 25.3385 #BP
    if band == 'RP': ZP = 24.7479 #RP
    return 10 ** ((ZP - magnitudes) / 2.5)

def flux_to_mag(fluxes, band):
    if band == 'G':  ZP = 25.6874  #G
    if band == 'BP': ZP = 25.3385 #BP
    if band == 'RP': ZP = 24.7479 #RP
    return -2.5 * np.log10(fluxes) + ZP
    
def log_normal(mean_log_a, sigma_log_a, num_samples):
    log_a_samples = np.random.normal(mean_log_a, sigma_log_a, num_samples)
    return log_a_samples

def bimodal_log_normal(mean_log_a1, sigma_log_a1, mean_log_a2, sigma_log_a2, num_samples, weight1=0.5):
    # Determine how many samples to draw from each distribution
    num_samples_1 = int(num_samples * weight1)
    num_samples_2 = num_samples - num_samples_1  # Remaining samples for the second distribution

    # Generate samples from the first log-normal distribution
    log_a_samples_1 = np.random.normal(mean_log_a1, sigma_log_a1, num_samples_1)
    
    # Generate samples from the second log-normal distribution
    log_a_samples_2 = np.random.normal(mean_log_a2, sigma_log_a2, num_samples_2)

    # Combine the two sets of samples
    combined_log_a_samples = np.concatenate([log_a_samples_1, log_a_samples_2])

    return combined_log_a_samples

def generate_powerlaw_data(mean_log_a, sigma_log_a, alpha, lower_value, upper_value, num_samples):
    # Generate uniform random numbers
    
    weight=0.2
    num_samples_gaussian = int(weight*num_samples)
    num_samples_powelaw  = int((1-weight)*num_samples)
    
    log_a_samples = np.random.normal(mean_log_a, sigma_log_a, num_samples_gaussian)

    random_values = np.random.uniform(0,1,num_samples_powelaw)
    
    # Apply the inverse CDF transformation to get power-law distributed samples
    powerlaw_samples = ((upper_value**(alpha+1) - lower_value**(alpha+1)) * random_values + lower_value**(alpha+1))**(1/(alpha+1))
    
    values_a = np.concatenate([log_a_samples, np.log10(powerlaw_samples)])
    
    return values_a

def generate_eccentricities(num_points):

    # Generate uniform random numbers between 0 and 1
    u = np.random.uniform(0, 1, num_points)
    
    # Apply the inverse CDF to get eccentricities
    eccentricities = u**1.2
    
    return eccentricities

def generate_physical_separation(semi_major_axis, eccentricity):

    inclination_deg = np.random.uniform(0,180,1)
    major_axis_angle_deg = np.random.uniform(0,360,1)
    # Convert inclination and major axis angle to radians
    inclination_rad = np.radians(inclination_deg)
    major_axis_angle_rad = np.radians(major_axis_angle_deg)
    
    # Generate random true anomalies (nu) uniformly distributed between 0 and 2*pi
    true_anomaly = np.random.uniform(0, 2 * np.pi, 1)
    
    # Calculate the physical separation r at each true anomaly
    r = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(true_anomaly))
    
    # Calculate the projected separation, accounting for the inclination and major axis orientation
    r_projected = r * np.sqrt(np.cos(true_anomaly + major_axis_angle_rad)**2 + 
                              np.sin(true_anomaly + major_axis_angle_rad)**2 * np.cos(inclination_rad)**2)
    
    return r_projected

def logsemimajoraxis(mass):
    A = np.zeros(mass.shape[0])

    w = mass<=0.1
    A[w] = 10**log_normal(np.log10(4.5), 0.5, mass[w].shape[0])

    w = (mass>0.1)&(mass<0.7)
    A[w] = 10**log_normal(np.log10(5.3), 1.3, mass[w].shape[0])

    w = (mass>=0.7)&(mass<1.5)
    A[w] = 10**log_normal(np.log10(45), 2.3, mass[w].shape[0])

    w = (mass>=1.5)&(mass<5)
    A[w] = 10**bimodal_log_normal(np.log10(0.1), 0.5, np.log10(350), 1.2, num_samples=mass[w].shape[0], weight1=0.35)

    w = (mass>=5)&(mass<16)
    A[w] = 10**bimodal_log_normal(np.log10(0.1), 0.5, np.log10(350), 1.2, num_samples=mass[w].shape[0], weight1=0.35)

    w = mass>=16
    A[w] = 10**generate_powerlaw_data(np.log10(0.1), 0.4, -1.1, 0.1, 10**5, num_samples=mass[w].shape[0])

    return A

def angular_separation_au_pc(physical_separation_au, distance_pc):
    
    # 1 parsec = 206,265 AU, so convert the distance from pc to AU
    distance_au = distance_pc * 206265
    
    # Calculate angular separation in arcseconds
    theta_arcseconds = (physical_separation_au / distance_au) * 206265
    
    return theta_arcseconds