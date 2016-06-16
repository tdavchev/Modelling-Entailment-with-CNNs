from sys import argv

script, filename, model = argv

target = open(filename, 'r')

model = int(model)
idx = -1
# skip the first row
first = 1
for line in target:
	if first < 1:
		elements = line.split('\t')
		if elements[0]=='entailment':
			idx = 0
		elif elements[0]=='neutral':
			idx = 1
		else:
			idx = 2
		if model == 1:
			print("{0}\t{1} {2}".format(idx,elements[5],elements[6]))
		elif model == 21: # record only first sentence and the element
			print("{0}\t{1}".format(idx,elements[5]))
		elif model == 22:
			print("{0}\t{2}".format(idx,elements[6]))
	else:
		first = 0

target.close()
