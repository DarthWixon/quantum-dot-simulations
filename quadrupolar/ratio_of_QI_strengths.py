import sys
sys.path.append("/home/will/Documents/phd/research/simulations/common_modules/")
import numpy as np
import matplotlib.pyplot as plt

graphing = True
fun_graphs = True

def QuadrupolarRatioCalculator(spin):
	return (2*spin - 1)*spin/(2*spin**2 - 3*spin + 1)

if graphing:
	max_spin = 10
	spin_range = np.arange(1.5, max_spin, 0.5)
	ratio_array = np.zeros(len(spin_range))

	for s, spin in enumerate(spin_range):
		ratio_array[s] = QuadrupolarRatioCalculator(spin)

	if fun_graphs:
		plt.xkcd()

	plt.scatter(spin_range, ratio_array)
	plt.xlabel('Spin')
	plt.ylabel('Ratio of QI Strength of Spin I to I-1/2')
	plt.title('How Much Does the QI Strength Change As Spin Increases?')
	plt.show()

if not graphing:
		
	base_spin = 1.5
	ratio_change = 100 # starter value, it will change quickly
	while abs(ratio_change) >= 10: # a percentage change
		ratio_1 = QuadrupolarRatioCalculator(base_spin)
		ratio_2 = QuadrupolarRatioCalculator(base_spin + 0.5)

		ratio_change = (ratio_2 - ratio_1)/ratio_2 * 100
		base_spin = base_spin + 0.5
		if base_spin > 100:
			break

	print('Term Ratio: {:.3f}'.format(ratio_2))
	print('Percentage Change: {:.3f}%'.format(ratio_change))
	print('Spin value: {:.1f}'.format(base_spin-0.5)) # undo the change in spin

