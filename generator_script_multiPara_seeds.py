import random

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

input_lengths = [10, 20, 30, 40, 50]
# input_lengths = [10]

N_values = list(range(5, 21, 5))
# N_values = [10]

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
	
swarm_file = open('swarm_script_multiParaSet_seeds.sh', 'w')

for record in selected_patients:

	for data_seed in random_data_seeds:

		for res_seed in random_res_seeds:

			for input_len in input_lengths:

				for N in N_values:

					for regime_id in range(len(parameter_regimes)):

						cmd = f'python3 DynML_selectedPatients_lead_I_online_detection-PCA-w1-10-multiParaSet_seeds.py {record} {data_seed} {res_seed} {input_len} {N} {regime_id}\n'
						swarm_file.write(cmd)

swarm_file.close()
