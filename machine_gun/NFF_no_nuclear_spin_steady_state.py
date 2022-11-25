import numpy as np

# set up ranges for variables, will look up reasonable values
# gonna have us a big nested for loop.....
w_range = np.linspace(0, 1, 2)
t_range = np.linspace(0, 1, 2)
q0_range = np.linspace(0, 1, 10)
phi_range = np.linspace(0, 2*np.pi, 10)

# create matrix describing the process as a superopertor
def MatrixCreator(w, t, q0, phi):
	q = q0*np.exp(1j*phi)
	qstar = q0*np.exp(-1j*phi)

	M = np.array([[np.cos(w*t)**2, 1j*qstar*np.cos(w*t)*np.sin(w*t), -1j*q*np.cos(w*t)*np.sin(w*t), 0.5*(-1 + q0**2) + q0**2 * np.sin(w*t)**2],
	[1j*np.cos(w*t)*np.sin(w*t), qstar*np.cos(w*t)**2, q*np.sin(w*t)**2, -1j*q0**2 * np.cos(w*t)*np.sin(w*t)],
	[-1j*np.cos(w*t)*np.sin(w*t), qstar*np.sin(2*t)**2, q*np.cos(w*t)**2, 1j*q0**2 * np.cos(w*t)*np.sin(w*t)],
	[np.sin(w*t)**2, -1j*qstar*np.cos(w*t)*np.sin(w*t), 1j*q*np.cos(w*t)*np.sin(w*t), 0.5*(-1 + q0**2) + q0**2 * np.cos(w*t)**2]])

	return M

def AllTheRanges(w_range, t_range, q0_range, phi_range):
	for w in w_range:
		for t in t_range:
			for q0 in q0_range:
				for phi in phi_range:
					M = MatrixCreator(w, t, q0, phi)
					eigvals, eigvectors = np.linalg.eig(M)

					for i in range(len(eigvals)):
						if 0.9999 < eigvals.real[i] < 1.0001:
							if abs(eigvals.imag[i] < 1e-10):
								eigenvalue_1_index = i
								
					print('w, t, q0, phi = {}, {}, {}, {}'.format(w,t,q0,phi))
					print('Steady state = {}'.format(eigvectors[eigenvalue_1_index]))


def FixedwAndt(w, t, q0_range, phi_range):
	storage = np.zeros((len(q0_range), len(phi_range), 3)) # 3 because I store q0, phi and the truth value
	for q0_index, q0 in enumerate(q0_range):
		for phi_index, phi in enumerate(phi_range):
			M = MatrixCreator(w, t, q0, phi)
			eigvals, eigvectors = np.linalg.eig(M)
			# print eigvals

			for i in range(len(eigvals)):

				if 0.9999 < eigvals.real[i] < 1.0001:
					if abs(eigvals.imag[i] < 1e-10):
						eigValue_1_exists = True
						eigenvalue_1_index = i
				else:
					eigValue_1_exists = False
			# if eigValue_1_exists:					
			# 	print 'w, t, q0, phi = {}, {}, {}, {}'.format(w,t,q0,phi)
			# 	print 'Steady state = {}'.format(eigvectors[eigenvalue_1_index])
			# else:
			# 	print 'No eigenvalue of 1 for these parameters. w, t, q0, phi = {}, {}, {}, {}'.format(w,t,q0,phi)

			storage[q0_index, phi_index] = [q0, phi, eigValue_1_exists]
	print(storage.shape)
	print(storage[:,:,2])

# Greilich paper values, from Greilich, A., et al. "Nuclei-induced frequency focusing of electron spin coherence." Science 317.5846 (2007): 1896-1899.
w = 300e9 # 300 Ghz
t = 13.2e-9 # 13.2 nS

FixedwAndt(w, t, q0_range, phi_range)



