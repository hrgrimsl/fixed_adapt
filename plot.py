import pylab
import numpy as np
import math

infile = 'cheap.dat'
read = open(infile, 'r')
E = []
error = []
log_error = []

for i in read.readlines():
    line = i.split()
    E.append(float(line[0]))
    error.append(float(line[1]))
    log_error.append(math.log10(float(line[1])))
log_error = np.array(log_error)
log_error[0:10] = log_error[0:10][::-1]
x = [.1*(i-10) for i in range(0, len(log_error))]
pylab.xlabel('Displacement from equilibrium ($\AA$)')
pylab.ylabel('$\log(E-E_{FCI})$ (log(kcal/mol))')
pylab.title('LiH PES from Fixed ADAPT')
pylab.plot(x, np.array(log_error))
pylab.savefig('plot.png')
pylab.show()

