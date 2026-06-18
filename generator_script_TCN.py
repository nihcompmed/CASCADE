import random

selected_patients = [
    '215','213','209','203','210','116','222','233',
    '118','223','221','214','200','228','201','208',
    '119','207','106'
]

random_data_seeds = [17]

random_res_seeds = [1107, 1249, 1388, 1523, 1697, 1841, 1999, 2134, 2288, 2456]

input_lengths = [10, 20, 30, 40, 50]

batch_sizes = [64, 128, 256]

hidden_dims = [100, 200, 300]

jobs = []

for record in selected_patients:
    for data_seed in random_data_seeds:
        for res_seed in random_res_seeds:
            for input_len in input_lengths:
                for batch_size in batch_sizes:
                    for hidden_dim in hidden_dims:

                        cmd = (
                            f'python3 '
                            f'TCN_selectedPatients_lead_I_online_detection-sequential-updating-PCA.py '
                            f'{record} {data_seed} {res_seed} {input_len} {batch_size} {hidden_dim}'
                        )

                        jobs.append(cmd)

# 🔀 Shuffle all jobs
random.shuffle(jobs)

# Write shuffled jobs to swarm file
with open('swarm_script_TCN.sh', 'w') as swarm_file:
    for job in jobs:
        swarm_file.write(job + '\n')