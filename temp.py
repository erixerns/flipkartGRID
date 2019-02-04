lims = (0, 10)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
ax1.add_patch(
    patches.Rectangle((0, 0), width, height))
plt.ylim(lims)
plt.xlim(lims)