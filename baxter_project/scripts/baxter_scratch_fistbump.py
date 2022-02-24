
import sympy
import sympybotics
import math
import numpy
import baxter_left_kinematics as blk

#pi = math.pi

#rbtdef = sympybotics.RobotDef('Baxter Thing 1', # robot name
#							#	alpha, 		a, 		d, theta
#							  [('-pi/2', 0.069, 0.27035, 'q'),
#							   ( 'pi/2',   0.0,     0.0, 'q+pi/2'),
#							   ('-pi/2', 0.069, 0.36435, 'q'),
#							   ( 'pi/2',   0.0,     0.0, 'q'),
#							   ('-pi/2',   0.1, 0.37429, 'q'),
#							   ( 'pi/2',   0.0,     0.0, 'q'),
#							   (      0,   0.0,0.229525, 'q')],
#							   dh_convention='standard')
#rbtdef.frictionmodel = {'viscous'}
#rbtdef.gravityacc = sympy.Matrix([0.0, 0.0, -9.81]) # optional, this is the default value

#rbt = sympybotics.RobotDynCode(rbtdef, verbose=True)

#Jsym = rbt.kin.J[-1]

# get the numerical version like this
# or generate code
#J = expr.subs([(x, 2), (y, 4), (z, 0)])



J = numpy.asmatrix(blk.J[6](q))
