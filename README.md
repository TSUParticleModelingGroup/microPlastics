# microPlastics

To-Do:
Look into disabling terminal until stir/shake needs input

Add a time innterval for shake/stir

When we start the simulation, we stir at 100 rpm for 1 minute, then we bring it to 50 rpm for 5 minutes, then the force function begins to work

Figure out how to make the microplastics stick to polymers and then sink once they are denser than the liquid after combining

adsorption rate function
fsp = ksp * nm * np
adsorption rate = collision frequency constant * concentration of microplastics * conc. of polymers

nm = 10 uL / 500 mL of sample liquid
np = 0.5 g/ 500 mL of sample = 0.5E9 ug/mL

ksp = [((2*kb*T)/3u) * (2 + dp/dm + dm/dp) + G/6 * (dm + dp)^3]
kb = 1.380649E-23 m^2 kg s^-2 k^-1 = 1.380649E7 um^2*ug*s^-2*k^-1
Abs Temp(T) = 21.9 C  = 295.05 k

dm = diameter of microplastic
dp = polymer diameter
G = shear rate = TBD(estimate)
u  = 0.013 Pa*s

Add polymer diameter range
Min = TBD (experiment)
Max = TBD (experiment)

UNITS- total runtime for this experiment is at least 6 minutes -> 360 seconds -> 360,000 milliseconds

For the container force, the forces are not enough on the larger particles, and too much for the smalleer particles, what if we changed it to be based on size?
Same goes for stirring if we can
