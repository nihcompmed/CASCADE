import random

# Generated patients: p00000 to p00039
patients_generated = [f"p{idx:05d}" for idx in range(41) if idx != 37]

# Manual patients
patients_manual = [
    'p00019', 'p00084', 'p00173', 'p00190', 'p00206',
    'p00214', 'p00244', 'p00261', 'p00283', 'p00284', 'p00380'
]

# Combine and remove duplicates
selected_patients = sorted(list(set(patients_generated + patients_manual)))

# random_data_seeds = [17, 83, 142, 256, 399, 512, 678, 745, 901, 1023]
random_data_seeds = [17]

random_res_seeds = [1107, 1249, 1388, 1523, 1697, 1841, 1999, 2134, 2288, 2456]


input_lengths = [10, 20, 30, 40, 50]

batch_sizes = [64, 128, 256]

hidden_dims = [100, 200, 300]

swarm_file = open('swarm_script_MLP_icentiaData.sh', 'w')

for record in selected_patients:

	for data_seed in random_data_seeds:

		for res_seed in random_res_seeds:

			for input_len in input_lengths:

				for batch_size in batch_sizes:

					for hidden_dim in hidden_dims:

						cmd = f'python3 MLP_icentia11kdata_selectedPatients_PCA_online.py {record} {data_seed} {res_seed} {input_len} {batch_size} {hidden_dim}\n'
						swarm_file.write(cmd)

swarm_file.close()
