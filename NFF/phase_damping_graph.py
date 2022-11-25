import numpy
import matplotlib.pyplot as pyplot

prob = numpy.linspace(0,1,1000)
gamma = (1 + numpy.sqrt(1 - prob))/2

# pyplot.xkcd()

pyplot.figure()
pyplot.plot(prob, gamma)
pyplot.xlabel('Photon Scattering Probability')
pyplot.ylabel('Phase Damping Parameter')
pyplot.xlim(0, 1)
pyplot.ylim(0, 1.01)

pyplot.axhline(0.5, linestyle = 'dotted')

pyplot.show()
