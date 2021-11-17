import os
import numpy as np
import random

path = './MultiDigitMNIST-master/datasets/multimnist'


os.mkdir("./data")
os.mkdir("./data/train")
os.mkdir("./data/val")
os.mkdir("./data/test")

train = []
test = []
val = []

for i in [0,2,4,6,8]:
	for j in [1,3,5,7,9]:
		
		trainpath = path + '/train/' + str(i) + str(j) + '/'
		valpath = path + '/val/' + str(i) + str(j) + '/'
		testpath = path + '/test/' + str(i) + str(j) + '/'
		
		if(os.path.isdir(trainpath)):
			
			ll = os.listdir(trainpath)
			ll = [trainpath + s for s in ll]
			random.shuffle(ll)
			ll = ll[0:1000]
			train.extend(ll)
			
			
		elif(os.path.isdir(testpath)):
		
			ll = os.listdir(testpath)
			ll = [testpath + s for s in ll]
			random.shuffle(ll)
			ll = ll[0:1000]
			test.extend(ll)
		
		else:
			ll = os.listdir(valpath)
			ll = [valpath + s for s in ll]
			random.shuffle(ll)
			ll = ll[0:1000]
			val.extend(ll)
	


for i in range(len(train)):
	
	
	file_to_move = train[i]
	command = "cp " + file_to_move + " ./data/train"
	os.system(command)
	
	#Test
	if(i < 5000):
		file_to_move = test[i]
		command = "cp " + file_to_move + " ./data/test"
		os.system(command)
	#Train
	if(i < 4000):
		file_to_move = val[i]
		command = "cp " + file_to_move + " ./data/val"
		os.system(command)
		
		

	if i == 5000:
		print("Moved 14000 so far...\n")

	if i == 10000:
		print("Moved 19000 so far, almost done...\n")
		
print("Done\n")























