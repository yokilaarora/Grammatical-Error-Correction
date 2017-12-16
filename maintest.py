from train import *
import torch
import numpy as np
from utils import *
import torch.utils.data as data_utils
from torch.autograd import Variable
import os

# Hyper parameters
batch_size = 15
num_epochs = 35      
learning_rate = 0.02 

# Parameters for NN model
V_s = 33820
V_t = 6643
H1 = 1000
H2 = 500
H3 = 1000

# Variables used to save the model 
start_epoch = 0
best_test_acc = 10

resume_from_file = True     #True if trained model, resumes from file
resume_file = 'model_best.pth.tar'
resumed_file = False

# Pre-process the data to get BOW representation for training and test sets
x = bag_of_words("conll14st-preprocessed","ann_file_new")
y = target_bag_of_words("ann_file_new")
xtest = bag_of_words("conllFile","test_ann_file_new")
ytest = target_bag_of_words("test_ann_file_new")

# Load data and divide into batches
print("Loading training data...")
train_dataset = data_utils.TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.from_numpy(np.asarray(y, dtype=np.float32)))
train_loader = data_utils.DataLoader(train_dataset,batch_size,shuffle = True, drop_last=True)

print("Loading test data...")
test_dataset = data_utils.TensorDataset(torch.from_numpy(np.asarray(xtest, dtype=np.float32)), torch.from_numpy(np.asarray(ytest, dtype=np.float32)))
test_loader = data_utils.DataLoader(test_dataset,batch_size,shuffle = True, drop_last=True)

# Initialize model with parameters
model = Model(V_s, H1, H2, H3, V_t)

for param in model.parameters():
    param.requires_grad = True

# Use gradient descent to update parameters
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

training_error= []
test_error = []

print('Running model...')
print('Batch Size: ',batch_size)
print('learning rate: ', learning_rate)
print('Size of hidden layer: ', H1)

if resume_from_file:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_file, checkpoint['epoch']))
        resumed_file = True
    else:
        print("=> no checkpoint found at '{}'".format(resume_file))

if not resume_from_file:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for x_, y_ in train_loader:
            x_, y_ = Variable(x_), Variable(y_)
            # Forward pass: compute predicted y by passing x to the model. 
            y_pred = model(x_)

            # Zero the gradients before running the backward pass.
            optimizer.zero_grad()

            # Compute loss. We pass Variables containing the predicted and true
            # values of y, and the loss function returns a Variable containing the loss.
            loss = torch.nn.functional.binary_cross_entropy(y_pred, y_)
            running_loss += loss.data[0]

            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model. 
            loss.backward()

            # Update the weights using gradient descent.
            optimizer.step()

        print('epoch [%d] loss: %.10f' % (epoch + 1, running_loss/1455))
        training_error.append(running_loss/1455)
        running_loss = 0.0

        # Test model
        count = 0
        runningtest_loss = 0.0
        for xtest_, ytest_ in test_loader:
            count = count+1
            xtest_, ytest_ = Variable(xtest_), Variable(ytest_)
            ytest_pred = model(xtest_)
            loss = torch.nn.functional.binary_cross_entropy(ytest_pred, ytest_)
            runningtest_loss += loss.data[0]
        
        print('Net Test Loss: %.10f' %(runningtest_loss/count))
        test_error.append(runningtest_loss/count)

        te = runningtest_loss/count
        is_best = te < best_test_acc
        best_test_acc = min(te,best_test_acc)

        # Save model
        if epoch % 10 == 0 or epoch == num_epochs-1:
            save_checkpoint({
                'epoch': start_epoch+epoch + 1,
                'state_dict': model.state_dict(),
                'best_test_acc': best_test_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
            print('Saved at epoch: ', epoch)
else:
    # After training, evaluate and check the model on test cases
    batch_randomtest = 1    #randomnly gives one sentence input output pair
    c = 0;
    t_loader = data_utils.DataLoader(test_dataset,batch_randomtest,shuffle = False, drop_last=True)

    # Trained SMT model, there are 5 predictions (lines) for each test case
    smt_file = open("trained_smt",'r')
    # Add all predictions to the list smt_preds
    smt_preds = []
    for line in smt_file:
        smt_preds.append(line)

    thresh = 0.03 #the threshold probability value above which the classifier output is considered to be one
    
    R, P, F05, e_count, g_count, common_count = 0,0,0,0,0,0
    print('Threshold taken for prediction:',thresh)
    
    for x_sample, y_sample in t_loader:
        print('Iteration:',c+1)
        xsample_, ysample_ = Variable(x_sample), Variable(y_sample)
        ypred_ = model(xsample_)        # hypothesis in Variable form
        y_pred = ypred_.data.numpy()    # hypothesis in array form
        x_sample = xsample_.data.numpy()
        y_sample = ysample_.data.numpy()
        x_s = x_sample[0]
        y_s = y_sample[0]
        y_p = y_pred[0]
        smt_y_pred = smt_preds[c-1:c-1+6]     # 5 SMT model predictions

        max_prob = 0
        best_pred = []
        for pred in smt_y_pred:
            prob = 0
            for w in range(len(pred)):
                if(prob==0):
                    prob = y_p[w]
                else:
                    prob = prob*y_p[w]
            if(prob>max_prob):
                max_prob = prob
                best_pred = pred

        # Define vocabularies to extract words
        VS = list(get_dict("conll14st-preprocessed","conllFile"))
        VT = list(target_get_dict("ann_file_new","test_ann_file_new"))
        VS_dict = {v: k for v, k in enumerate(VS)}  # Source Vocab dictionary
        VT_dict = {v: k for v, k in enumerate(VT)}  # Target Vocab dictionary

        source_words = []
        target_words = []
        hypothesis = []
        a = []

        # Find bag of words for the input sentence
        for ws in range(len(VS)):
            if x_s[ws] == 1:
                source_words.append(VS_dict[ws])

        for wt in range(len(VT)):
            if y_s[wt] == 1:
                a.append(wt)
                target_words.append(VT_dict[wt])
            if best_pred[wt] >= thresh:   
                hypothesis.append(VT_dict[wt])
            
        common_words = set(hypothesis).intersection(target_words)
        ec = len(hypothesis)
        eg = len(target_words)
        ecommon = len(common_words)
        e_count = e_count + ec      # number of predicted corrections
        g_count = g_count + eg      # number of gold-edit corrections
        common_count = common_count + ecommon #common/-Intersection
        # F0.5 score for a set of 1 sentence only
        if e_count != 0 and g_count != 0:
            R = common_count/g_count
            P = common_count/e_count
            if (R+ 0.5*0.5*P) != 0:
                F05 = ((1+0.5*0.5)*R*P)/(R+0.5*0.5*P)
            else:
                print('F05 had denominator 0 for this calculation, so not calculated')
        else:
            print('Either one of edit count is zero')

        print('Words of Source Sentence',)
        print(source_words)
        print('Words of Target Sentence')
        print(target_words)
        print('Model generated sentence')
        print(hypothesis)
        print('No. of desired/gold-standard corrections for this iteration and in total till now')
        print(eg,g_count)
        print('No. of model\'s corrections for this iteration and in total till now')
        print(ec,e_count)
        print('Length of intersection of prediction and desired in this iteration and till now:')
        print(ecommon,common_count)
        print('And the common words are:',common_words)
        print('R, P and F0.5 score, respectively:',R,P,F05)

        c = c+1

