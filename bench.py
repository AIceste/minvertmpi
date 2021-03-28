import subprocess
import sys

binaries = ['./tp3_sequential', './tp3_parallel']
test_count = 1
sizes = [4, 42, 420, 926, 2100]
thread_counts = [8, 7, 6, 5, 4, 3, 2, 1]
seed = 0

tests_seq = [{'bin': './tp3_sequential', 'size': s, 'durations': [], 'errors': []}
	for s in sizes
]

tests_para = [{'bin': './tp3_parallel', 'size': s, 'thread_count': c, 'durations': [], 'errors': []}
	for s in sizes
	for c in thread_counts
]
for i in range(test_count):
	for test in tests_seq:
		sorties = (subprocess.run([test['bin'], str(test['size']), str(seed)], text=True,
									stderr=subprocess.PIPE).stderr).split("\n")
		test['durations'] += [float((sorties[2].split(" "))[-1])]
		test['errors'] += [float((sorties[3].split(" "))[-1])]
		with open(sys.argv[1], 'a') as f:
			f.write(str(test) + '\n')

for i in range(test_count):
	for test in tests_para:
		sorties = (subprocess.run(['mpirun', '-np', str(test['thread_count']),
								test['bin'], str(test['size']), str(seed)], text=True,
								stderr=subprocess.PIPE).stderr).split("\n")
		test['durations'] += [float((sorties[2].split(" "))[-1])]
		test['errors'] += [float((sorties[3].split(" "))[-1])]
		with open(sys.argv[1], 'a') as f:
			f.write(str(test) + '\n')

