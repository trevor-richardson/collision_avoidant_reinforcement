import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


zero_action_dir = "/home/trevor/coding/robotic_pain/collision_avoidant_reinforcement/machine_learning/results/baselines/0_action/valbaseline_results.npy"
one_action_dir = "/home/trevor/coding/robotic_pain/collision_avoidant_reinforcement/machine_learning/results/baselines/1_action/valbaseline_results.npy"
two_action_dir = "/home/trevor/coding/robotic_pain/collision_avoidant_reinforcement/machine_learning/results/baselines/2_action/valbaseline_results.npy"
three_action_dir = "/home/trevor/coding/robotic_pain/collision_avoidant_reinforcement/machine_learning/results/baselines/3_action/valbaseline_results.npy"
four_action_dir = "/home/trevor/coding/robotic_pain/collision_avoidant_reinforcement/machine_learning/results/baselines/4_action/valbaseline_results.npy"
rand_action_dir = "/home/trevor/coding/robotic_pain/collision_avoidant_reinforcement/machine_learning/results/baselines/rand_action/valbaseline_results.npy"

zero = np.load(zero_action_dir)
one = np.load(one_action_dir)
two = np.load(two_action_dir)
three = np.load(three_action_dir)
four = np.load(four_action_dir)
rand = np.load(rand_action_dir)

zero = zero[:,1]
one = one[:,1]
two = two[:,1]
three = three[:,1]
four = four[:,1]
rand = rand[:,1]

data = [zero, one, two, three, four, rand]
N = 10
fig, ax1 = plt.subplots(figsize=(10, 6))
fig.canvas.set_window_title('A Boxplot Example')
fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)


plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

# Hide these grid behind plot objects
ax1.set_axisbelow(True)
ax1.set_title('Baseline Models Dodge Performance Over 50 Simulation Runs')
ax1.set_xlabel('Distribution')
ax1.set_ylabel('Value')

# Now fill the boxes with desired colors
boxColors = ['darkkhaki', 'royalblue', 'cyan', 'magenta', 'green', 'yellow']
numBoxes = 6
medians = list(range(numBoxes))
for i in range(numBoxes):
    box = bp['boxes'][i]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = np.column_stack([boxX, boxY])
    # Alternate between Dark Khaki and Royal Blue
    k = i % 2
    boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
    ax1.add_patch(boxPolygon)
    # Now draw the median lines back over what we just filled in
    med = bp['medians'][i]
    medianX = []
    medianY = []
    for j in range(2):
        medianX.append(med.get_xdata()[j])
        medianY.append(med.get_ydata()[j])
        ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
    ax1.plot([np.average(med.get_xdata())], [np.average(data[i])],
             color='w', marker='*', markeredgecolor='k')

# Set the axes ranges and axes labels
ax1.set_xlim(0.5, numBoxes + 0.5)
top = 40
bottom = -5
ax1.set_ylim(bottom, top)

randomDist = ['Action 0', 'Action 1', 'Action 2', 'Action 3',
               'Action 4', 'Rand Action']

ax1.set_xticklabels(np.repeat(randomDist, 1),
                    rotation=45, fontsize=8)

# Due to the Y-axis scale being different across samples, it can be
# hard to compare differences in medians across the samples. Add upper
# X-axis tick labels with the sample medians to aid in comparison
# (just use two decimal places of precision)


pos = np.arange(numBoxes) + 1
upperLabels = [str(np.round(s, 2)) for s in medians]
weights = ['bold', 'semibold']
for tick, label in zip(range(numBoxes), ax1.get_xticklabels()):
    # k = tick % 2
    ax1.text(pos[tick], top - (top*0.05), upperLabels[tick],
             horizontalalignment='center', size='x-small', weight=weights[k],
             color=boxColors[k])

# Finally, add a basic legend
fig.text(0.16667, 0.08, str(N) + 'Action 0',
         backgroundcolor=boxColors[0], color='black', weight='roman',
         size='x-small')
fig.text(0.3334, 0.08, 'Action 1',
         backgroundcolor=boxColors[1],
         color='white', weight='roman', size='x-small')
fig.text(0.66667, 0.08, 'Action 2',
         backgroundcolor=boxColors[2],
         color='white', weight='roman', size='x-small')
fig.text(0.83334, 0.08, 'Action 3',
         backgroundcolor=boxColors[3],
         color='white', weight='roman', size='x-small')
fig.text(0.1, 0.045, 'Action 4',
         backgroundcolor=boxColors[4],
         color='white', weight='roman', size='x-small')
fig.text(0.80, 0.045, 'Random Action',
         backgroundcolor=boxColors[5],
         color='white', weight='roman', size='x-small')
fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
         weight='roman', size='medium')
fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
         size='x-small')

plt.show()



# plt.boxplot(postings)
# plt.show()
