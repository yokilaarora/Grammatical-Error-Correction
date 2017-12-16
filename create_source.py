# Method to generate source file for SMT model
def sourceFile(f1,f2):
	source1 = open(f1,'r')
	source_out = open(f2,'w')
	count = 0;
	for line in source1:
		parts = line.split()
		print(count)
		if(len(parts)>0):
			if parts[0] == "S":
				parts.pop(0)
				output = ""
				source = parts
				count = 0
			elif parts[0] == "A":
				for i in range(len(source)):
					output = output+source[i]
					output = output + " "
				if count == 0:
					source_out.write("%s\n" %output)
					count = count +1
		
	source_out.close()

sourceFile("conll14st-preprocessed.m2","source_file.txt")
