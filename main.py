from dwave.system.samplers import DWaveSampler
from helpers.draw import plot_schedule
from dwave.system import TilingComposite, FixedEmbeddingComposite
from helpers.draw import plot_success_fraction
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np


sampler = DWaveSampler()
print("Connected to sampler", sampler.solver.name)
print("Maximum anneal schedule points: {}".format(sampler.properties["max_anneal_schedule_points"]))

annealing_range = sampler.properties["annealing_time_range"]
max_slope = 1.0 / annealing_range[0]
print("Annealing time range: {}".format(sampler.properties["annealing_time_range"]))
print("Maximum slope:", max_slope)

# Print the full anneal schedule
schedule = [[0.0, 0.0], [50.0, 0.5], [250.0, 0.5], [300, 1.0]]
print("Schedule: %s" % schedule)

# Plot the anneal schedule
plot_schedule(schedule, "Example Anneal Schedule with Pause")

schedule = [[0.0, 0.0], [12.0, 0.6], [12.8, 1.0]]
print("Schedule: %s" % schedule)

plot_schedule(schedule, "Example Anneal Schedule with Quench")

schedule=[[0.0, 0.0], [40.0, 0.4], [90.0, 0.4], [91.2, 1.0]]
print("Schedule: %s" % schedule)

plot_schedule(schedule, "Example Anneal Schedule with Pause and Quench")

h = {0: 1.0, 1: -1.0, 2: -1.0, 3: 1.0, 4: 1.0, 5: -1.0, 6: 0.0, 7: 1.0,
     8: 1.0, 9: -1.0, 10: -1.0, 11: 1.0, 12: 1.0, 13: 0.0, 14: -1.0, 15: 1.0}
J = {(9, 13): -1, (2, 6): -1, (8, 13): -1, (9, 14): -1, (9, 15): -1,
     (10, 13): -1, (5, 13): -1, (10, 12): -1, (1, 5): -1, (10, 14): -1,
     (0, 5): -1, (1, 6): -1, (3, 6): -1, (1, 7): -1, (11, 14): -1,
     (2, 5): -1, (2, 4): -1, (6, 14): -1}


tiled_sampler = TilingComposite(sampler, 1, 2, 4)

if tiled_sampler.num_tiles:
    sampler_embedded = FixedEmbeddingComposite(sampler, embedding=tiled_sampler.embeddings[0])
else:
    print("Unable to find two complete unit cells in QPU {}".format(sampler.solver.name))

runs = 1000
results = sampler_embedded.sample_ising(h, J,
                                        num_reads=runs,
                                        answer_mode='raw',
                                        label='Notebook - Anneal Schedule',
                                        annealing_time=100)

print("QPU time used:", results.info['timing']['qpu_access_time'], "microseconds.")


plt.hist(results.record.energy,rwidth=1,align='left',bins=[-21,-20,-19,-18,-17,-16,-15])
plt.show()


_, counts = np.unique(results.record.energy.reshape(1000,1), axis=0, return_counts=True)
print("Ground state probability: ", counts[0]/runs)

with open("files/saved_pause_results.json", "r") as read_file:
    saved_pause_success_prob = pd.read_json(read_file)


pause_plot = plot_success_fraction(saved_pause_success_prob,
                      "Success Fraction Using Pause for a Range of Anneal-Schedule Parameters",
                      "pause_duration")

#Update the standard anneal schedule parameters below:

anneal_time = 20.0
pause_duration = 20.0      # Must be greater than 0
pause_start = 0.3        # Must be between 0 and 1

#----------------------------------------------------------------
#Leave the code below to run the problem and display the results.
#----------------------------------------------------------------
schedule=[[0.0,0.0],[pause_start*anneal_time,pause_start],[pause_start*anneal_time+pause_duration, pause_start],[anneal_time+pause_duration, 1.0]]
runs=1000
results = sampler_embedded.sample_ising(h, J,
                anneal_schedule=schedule,
                num_reads=runs,
                answer_mode='raw',
                label='Notebook - Anneal Schedule')
success = np.count_nonzero(results.record.energy == -20.0)/runs
print("Success probability: ",success)

pause_plot["axis"].scatter([pause_start],[success], color="red", s=100)
pause_plot["figure"]

with open("files/saved_quench_results.json", "r") as read_file:
    saved_quench_success_prob = pd.read_json(read_file).replace(0, np.nan)

quench_plot = plot_success_fraction(saved_quench_success_prob,
                                    "Success Fraction Using Quench for a Range of Anneal-Schedule Parameters",
                                    "quench_slope")

#Update the standard anneal schedule parameters below

anneal_time = 50.0
quench_slope = 1.0      # Must be greater than 0
quench_start = 0.45      # Must be between 0 and 1

#----------------------------------------------------------------
#Leave the code below to run the problem and display the results.
#----------------------------------------------------------------
schedule=[[0.0,0.0],[quench_start*anneal_time,quench_start],[(1-quench_start+quench_slope*quench_start*anneal_time)/quench_slope, 1.0]]
runs=1000
results = sampler_embedded.sample_ising(h, J,
                anneal_schedule=schedule,
                num_reads=runs,
                answer_mode='raw',
                label='Notebook - Anneal Schedule')
success = np.count_nonzero(results.record.energy == -20.0)/runs
print("Success probability: ",success)

quench_plot["axis"].scatter([quench_start],[success], color="red", s=100)
quench_plot["figure"]

anneal_time = [10.0, 100.0, 300.0]
pause_duration = [10.0, 100.0, 300.0]

# Create list of start times
num_points = 5
s_low = 0.2
s_high = 0.6
pause_start = np.linspace(s_low, s_high, num=num_points)

success_prob = pd.DataFrame(index=range(len(anneal_time) * len(pause_duration) * len(pause_start)),
                            columns=["anneal_time", "pause_duration", "s_feature", "success_frac"],
                            data=None)
counter = 0

print("Starting QPU calls...")
QPU_time = 0.0
for anneal in anneal_time:
    for pause in pause_duration:
        for start in pause_start:
            schedule = [[0.0, 0.0], [start * anneal, start], [start * anneal + pause, start], [anneal + pause, 1.0]]
            runs = 1000
            results = sampler_embedded.sample_ising(h, J,
                                                    anneal_schedule=schedule,
                                                    num_reads=runs,
                                                    answer_mode='raw',
                                                    label='Notebook - Anneal Schedule')
            success_prob.iloc[counter] = {"anneal_time": anneal,
                                          "pause_duration": pause,
                                          "s_feature": start,
                                          "success_frac": np.count_nonzero(results.record.energy == -20.0) / runs}
            counter += 1
            QPU_time += results.info['timing']['qpu_access_time']
        print("QPU calls remaining: ", len(anneal_time) * len(pause_duration) * len(pause_start) - counter)

print("QPU calls complete using", QPU_time / 1000000.0, "seconds of QPU time.")

anneal_time = [10.0, 100.0, 300.0]
quench_slopes = [1.0, 0.5, 0.25]

# start times
num_points = 5
s_low = 0.2
s_high = 0.9
quench_start = np.linspace(s_low, s_high, num=num_points)

success_prob = pd.DataFrame(index=range(len(anneal_time) * len(quench_slopes) * len(quench_start)),
                            columns=["anneal_time", "quench_slope", "s_feature", "success_frac"],
                            data=None)
counter = 0

print("Starting QPU calls...")
QPU_time = 0.0
for anneal in anneal_time:
    for quench in quench_slopes:
        for start in quench_start:
            schedule = [[0.0, 0.0], [start * anneal, start], [(1 - start + quench * start * anneal) / quench, 1.0]]
            runs = 1000
            results = sampler_embedded.sample_ising(h, J,
                                                    anneal_schedule=schedule,
                                                    num_reads=runs,
                                                    answer_mode='raw',
                                                    label='Notebook - Anneal Schedule')
            success_prob.iloc[counter] = {"anneal_time": anneal,
                                          "quench_slope": quench,
                                          "s_feature": start,
                                          "success_frac": np.count_nonzero(results.record.energy == -20.0) / runs}
            counter += 1
            QPU_time += results.info['timing']['qpu_access_time']
        print("QPU calls remaining: ", len(anneal_time) * len(quench_slopes) * len(quench_start) - counter)

print("QPU calls complete using", QPU_time / 1000000.0, "seconds of QPU time.")