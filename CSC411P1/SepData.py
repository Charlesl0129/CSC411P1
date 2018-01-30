trainingSet_bracco = set()
validationSet_bracco = set()
testSet_bracco = set()
    
dataSet_bracco = set()
for i in range (120):
    dataSet_bracco.add('bracco'+str(i))
for i in range (71):
    trainingSet_bracco.add(dataSet_bracco.pop())
for i in range (11):
    validationSet_bracco.add(dataSet_bracco.pop())
for i in range (11):
    testSet_bracco.add(dataSet_bracco.pop())

trainingSet_gilpin = set()
validationSet_gilpin = set()
testSet_gilpin = set()
    
dataSet_gilpin = set()
for i in range (88):
    dataSet_gilpin.add('gilpin'+str(i))
for i in range (66):
    trainingSet_gilpin.add(dataSet_gilpin.pop())
for i in range (11):
    validationSet_gilpin.add(dataSet_gilpin.pop())
for i in range (11):
    testSet_gilpin.add(dataSet_gilpin.pop())
    
trainingSet_harmon = set()
validationSet_harmon = set()
testSet_harmon = set()
    
dataSet_harmon = set()
for i in range (139):
    dataSet_harmon.add('harmon'+str(i))
for i in range (71):
    trainingSet_harmon.add(dataSet_harmon.pop())
for i in range (11):
    validationSet_harmon.add(dataSet_harmon.pop())
for i in range (11):
    testSet_harmon.add(dataSet_harmon.pop())

trainingSet_baldwin = set()
validationSet_baldwin = set()
testSet_baldwin = set()
    
dataSet_baldwin = set()
for i in range (136):
    dataSet_baldwin.add('baldwin'+str(i))
for i in range (71):
    trainingSet_baldwin.add(dataSet_baldwin.pop())
for i in range (11):
    validationSet_baldwin.add(dataSet_baldwin.pop())
for i in range (11):
    testSet_baldwin.add(dataSet_baldwin.pop())

trainingSet_hader = set()
validationSet_hader = set()
testSet_hader = set()
    
dataSet_hader = set()
for i in range (141):
    dataSet_hader.add('hader'+str(i))
for i in range (71):
    trainingSet_hader.add(dataSet_hader.pop())
for i in range (11):
    validationSet_hader.add(dataSet_hader.pop())
for i in range (11):
    testSet_hader.add(dataSet_hader.pop())

trainingSet_carell = set()
validationSet_carell = set()
testSet_carell = set()
    
dataSet_carell = set()
for i in range (134):
    dataSet_carell.add('carell'+str(i))
for i in range (71):
    trainingSet_carell.add(dataSet_carell.pop())
for i in range (11):
    validationSet_carell.add(dataSet_carell.pop())
for i in range (11):
    testSet_carell.add(dataSet_carell.pop())

