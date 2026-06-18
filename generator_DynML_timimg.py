#!/usr/bin/env python
# coding: utf-8

# ==========================================================
# SWARM GENERATOR FOR TIMING EXPERIMENTS
# Matches timing_cascade.py argument order exactly:
#   python3 timing_cascade.py <record> <data_seed> <res_seed> <input_len> <N>
# ==========================================================

selected_patients = [
    '215','213','209','203','210','116','222','233',
    '118','223','221','214','200','228','201','208',
    '119','207','106'
]

# random_data_seeds = [17, 83, 142, 256, 399, 512, 678, 745, 901, 1023]
random_data_seeds = [17]

# random_res_seeds = [1107, 1249, 1388, 1523, 1697, 1841, 1999, 2134, 2288, 2456]
random_res_seeds = [1107, 1249, 1388, 1523, 1697]
# random_res_seeds = [1841, 1999, 2134, 2288, 2456]

# input_lengths = [10, 20, 30, 40, 50]
input_lengths = [10]

N_values = list(range(5, 51, 5))
# N_values = [10]

swarm_file = open('swarm_timing.sh', 'w')

for record in selected_patients:
    for data_seed in random_data_seeds:
        for res_seed in random_res_seeds:
            for input_len in input_lengths:
                for N in N_values:
                    cmd = (
                        'python3 timing_cascade.py '
                        '{} {} {} {} {}\n'.format(
                            record, data_seed, res_seed, input_len, N)
                    )
                    swarm_file.write(cmd)

swarm_file.close()

n_jobs = (len(selected_patients) * len(random_data_seeds) *
          len(random_res_seeds) * len(input_lengths) * len(N_values))

print("Generated swarm_timing.sh")
print("Total jobs        : {:,}".format(n_jobs))
print("  Patients        : {}".format(len(selected_patients)))
print("  Data seeds      : {}".format(random_data_seeds))
print("  Reservoir seeds : {}".format(random_res_seeds))
print("  Input lengths   : {}".format(input_lengths))
print("  N values        : {}".format(N_values))
print("\nEach job produces one CSV in results_timing/")
print("\nMake swarm log folder: mkdir swarm_logs_time")
print("\nSubmit with:")
print("  swarm -f swarm_timing.sh --time 04:00:00 -t 4 -g 1 --logdir swarm_logs_time")