from learn_recall_prefix import PrefixLearner
from sys import argv

lr = float(argv[1])
q = "While working on a scene for an action movie, a sound technician is given the task of changing the frequency of a gunshot to more accurately reflect the normal speed of sound. The gunshot came from an actor inside a car traveling 108 km/h, and it was recorded by a camera on a platform 200 meters away traveling at 72 km/h in the same direction. If the frequency of the gunshot is normally 800Hz, what is the perceived frequency which the camera picks up the gunshot at?"

pl = PrefixLearner("EleutherAI/gpt-neo-125m")
pl.learn_prefix(q, lr=lr)
