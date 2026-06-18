import random

# # selected_patients = [
# #     'p00019', 'p00084', 'p00173', 'p00190', 'p00206',
# #     'p00214', 'p00244', 'p00261', 'p00283', 'p00284', 'p00380'
# # ]
# # selected_patients = [f"p{idx:05d}" for idx in range(16)]
# selected_patients = [f"p{idx:05d}" for idx in range(16, 42) if idx != 19]

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
# input_lengths = [20]

# N_values = list(range(5, 51, 5))
# N_values = [10, 20, 30, 40, 50]
# N_values = [2, 5] + list(range(10, 51, 5))
N_values = list(range(2, 21, 2)) + list(range(25, 51, 5))

# ====== MULTIPLE PARAMETER REGIMES ======
parameter_regimes = [
    [(0.02, 0.07),  (0.02, 0.07),  (6.0, 7.0)],     # 0 far stable
    [(0.06, 0.11),  (0.06, 0.11),  (5.9, 6.6)],     # 1 moderately far
    [(0.09, 0.13),  (0.09, 0.13),  (5.8, 6.4)],     # 2 approaching
    [(0.11, 0.15),  (0.11, 0.15),  (5.7, 6.2)],     # 3 pre-edge
    [(0.13, 0.17),  (0.13, 0.17),  (5.6, 6.1)],     # 4 edge-ish
    [(0.15, 0.19),  (0.15, 0.19),  (5.5, 6.0)],     # 5 near edge
    [(0.17, 0.195), (0.17, 0.195), (5.45, 5.85)]    # 6 very near chaotic center
]

selected_regime_ids = [6] # only run the code for selected regime ids
	
swarm_file = open('swarm_script_DynML_multiParaSet_seeds_icentiaData.sh', 'w')

for record in selected_patients:

	for data_seed in random_data_seeds:

		for res_seed in random_res_seeds:

			for input_len in input_lengths:

				for N in N_values:

					# for regime_id in range(len(parameter_regimes)):
					for regime_id in selected_regime_ids:

						cmd = f'python3 DynML_icentia11kdata_selectedPatients_PCA_online.py {record} {data_seed} {res_seed} {input_len} {N} {regime_id}\n'
						swarm_file.write(cmd)

swarm_file.close()

