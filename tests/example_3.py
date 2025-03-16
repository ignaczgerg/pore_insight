
import numpy as np
import pore_insight
from pore_insight import pore_size_distribution as psd

# Two point Aimar methodology
rs = np.array([0.264,0.409,0.400,0.383,0.424,0.530,0.804,0.570]) # nm
rejections = np.array([22.67,60.00,67.40,66.90,72.09,74.16,100,100])/100 #%

r0, r1 = 4,3

a0_example = rs[r0] # nm
R0_example = rejections[r0]
a1_example = rs[r1] # nm
R1_example = rejections[r1]

twopoints = psd.TwoPointPSD(
    solute_radius_zero=a0_example,
    rejection_zero=R0_example,
    solute_radius_one=a1_example,
    rejection_one=R1_example
)

print("\nExample 3 - Two Points Method")

twopoints.find_pore_distribution_params()
print("log-normal Parameters",twopoints.lognormal_parameters)

twopoints.predict_rejection_curve(a = 0.264)
print("Retention prediction:",twopoints.prediction)