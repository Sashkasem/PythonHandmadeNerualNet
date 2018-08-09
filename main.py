import Net
import PIL
from mnist import MNIST
mndata = MNIST(r'''C:\Users\User\PycharmProjects\NeuronetHandmade\mnist_data\download''')
mndata.gz = True
images, labels = mndata.load_training()
to_upload = [0.0]*10
#file = open("TrainingData.txt", 'r')
#lines = file.readlines()
#topology = [int(x) for x in lines[0].split()]
topology = [784, 100, 10]
#file.close()
print("topology: ",topology)
myNet = Net.net(topology)
resultVals = [0.0]
for repeat in range(1):
    for i in range(len(images)):

        for q in range(len(images[i])):
            images[i][q]/=256.0 #scaling

        inputVals = images[i]
        #print("input: ",inputVals)
        to_upload[labels[i]] = 1.0
        outputVals = to_upload


        #print("output: ",outputVals)


        myNet.feedForward(inputVals)
        myNet.backProp(outputVals)
        myNet.getResults(resultVals)
        print("error:", myNet.getRecentAverageError(), " learning: ",i/len(images),"%")
        #print("result: ", resultVals)
        to_upload[labels[i]] = 0.0

    #print("error:", myNet.getRecentAverageError())
    successes = 0
    images_test, labels_test = mndata.load_testing()
    for i in range(len(images_test)):

        for q in range(len(images_test[i])):
            images_test[i][q] /= 256.0  # scaling

        inputVals = images_test[i]
        # print("input: ",inputVals)

        # print("output: ",outputVals)

        myNet.feedForward(inputVals)
        myNet.getResults(resultVals)
        max = 0
        indexmax = 0
        for f in range(len(resultVals)):
            if resultVals[f] > max:
                max = resultVals[f]
                indexmax = f

        print("result: ", indexmax, " right: ", labels_test[i])
        if indexmax == labels_test[i]:
            print("Yes")
            successes+=1
        else:
            print("No")
    print("successes = ",successes/len(images_test))
