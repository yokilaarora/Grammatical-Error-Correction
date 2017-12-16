from itertools import izip
f1 = "source_file.txt"
f2 = "target_file.txt"
source = open(f1,'r')
target = open(f2,'r')
#source_lines = 
for line1, line2 in izip(source,target):
	parts1 = line1.split()
	parts2 = line2.split()
	if(parts1[0]!=parts2[0]):
		print('-'*40)
		print(line1)
		print(line2)
		print('*'*40)
		assert(False)

