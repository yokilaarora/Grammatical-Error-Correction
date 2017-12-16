# Method to generate source file for SMT model
def targetFile(f1,f2):
	target1 = open(f1,'r')
	target_out = open(f2,'w')
	counter = 0
	pairs2_1 = []
	pairs2_2 = []
	words=[]
	for line in target1:
		parts = line.split()
		parts1= line.split("|||")
		parts2= parts1[0].split()

		if(len(parts) > 0):
			if parts[0] == "S":
				parts.pop(0)
				source = parts
			elif (parts[0] == "A"):
				counter=10
				target = source[0:int(parts2[1])] + [parts1[2]] + source[int(parts2[2]):]
				pairs2_1.append(int(parts2[1]))
				pairs2_2.append(int(parts2[2]))
				words.append(parts1[2])
				assert(len(pairs2_1)==len(pairs2_2))
				assert(len(pairs2_1)==len(words))
		elif(len(parts) == 0 and len(pairs2_1)>0):
			n = len(pairs2_1)
			for i in range(n-1,-1,-1):
				source = source[0:pairs2_1[i]] + [words[i]] + source[pairs2_2[i]:]
			
			output = ""
	
			for i in range(len(source)):
				output = output+source[i]
				output = output + " "
			if counter == 10:
				target_out.write("%s\n" %output)
			source = []
			pairs2_1 = []
			words = []
			pairs2_2 = []
			counter = 0

	target_out.close()

targetFile("conll14st-preprocessed.m2","target_file.txt")
