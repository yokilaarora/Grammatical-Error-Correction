'''
This file is for preprocessing input for Neural Network Model, and for saving the model.
There are different methods for source/input and target/output files, since the files 
have different formatting.
'''
import numpy as np
import re
import torch
import shutil

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # Method to save model parameters 
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def get_dict(f1,f2):
    # Inputs file f1, f2 and returns a sorted set of words present in either or both of the files
    f_1 = open(f1, 'r')
    f_2 = open(f2, 'r')
    words_list = []

    for line in f_1:
        parts = line.split()
        if len(parts)>1:
            words_list.append(parts[4])

    for line in f_2:
        parts = line.split()
        if len(parts)>1:
            words_list.append(parts[4])

    words_list = sorted(set(words_list))
    return words_list


def target_parse(file1):
    # Converts the annFile into a better format -> ann_file_new
    # which contains the <nid, pid, sid, start_token, end_token, error type, correction> columns
    f1 = open(file1, 'r')
    ann_file_new = open("test_ann_file_new",'w')
    for line in f1:
        parts = line.split()
        if(len(parts)>0):
            if(parts[0]=="<MISTAKE"):
                nid = re.findall(r'"([^"]*)"', parts[1])
                pid = re.findall(r'"([^"]*)"', parts[2])
                sid = re.findall(r'"([^"]*)"', parts[3])
                start_token = re.findall(r'"([^"]*)"', parts[4])
                end_token = re.findall(r'"([^"]*)"', parts[5])
                ann_file_new.write(nid[0] + " ")
                ann_file_new.write(pid[0] + " ")
                ann_file_new.write(sid[0] + " ")
                ann_file_new.write(start_token[0] + " ")
                ann_file_new.write(end_token[0] + " ")
            elif(line[0:6]=="<TYPE>"):
                type_ = line[6:len(parts[0])-7]
                ann_file_new.write(type_+" ")
            elif(line[0:12]=="<CORRECTION>"):
                corrected_word = line[12:len(line)-14]
                ann_file_new.write(corrected_word+"\n")
        else:
            ann_file_new.write("\n")
    ann_file_new.close()

def target_get_dict(f1,f2):
    # Inputs file f1, f2 and returns a sorted set of words present in either or both of the files
    f_1 = open(f1, 'r')
    f_2 = open(f2,'r')
    words_list = []

    for line in f_1:
        parts = line.split()
        if len(parts)>6:
            n_parts = len(parts)-6
            while(n_parts>0):
                words_list.append(parts[6+n_parts-1])
                n_parts -= 1

    for line in f_2:
        parts = line.split()
        if len(parts)>6:
            n_parts = len(parts)-6
            while(n_parts>0):
                words_list.append(parts[6+n_parts-1])
                n_parts -= 1

    words_list = sorted(set(words_list))

    return words_list

def target_bag_of_words(fn):
    # Returns bag of words representations for target (output/corrected) sentences
    f = open(fn, 'r')
    fn_target_train = "ann_file_new"
    fn_target_test = "test_ann_file_new"

    # create vocabulary from test and train data, both
    words_list = list(target_get_dict(fn_target_train,fn_target_test))
    words_list = {k: v for v, k in enumerate(words_list)} # convert list to dict
    n = len(words_list)
    
    A = []
    c = []
    l = np.zeros(n)
    ct = 0
    for line in f:
        parts = line.split()
        if len(parts)>6:
            n_parts = len(parts)-6
            while(n_parts>0):
                word = parts[6+n_parts-1]
                if(word in words_list):
                    index = words_list[word]
                    l[index] = 1
                else:
                    ct = ct+1
                n_parts -= 1
        elif(len(parts)<2):
            A.append(l)
            l = np.zeros(n)
            c.append(ct)
            ct = 0
    return A

def bag_of_words(fn, fn1):
    # Returns bag of words representations for source (input) sentences 
    f = open(fn, 'r')
    f1 = open(fn1, 'r')
    list_target = []
    for line in f1:
        parts = line.split()
        if(len(parts)>2):
            pid = parts[0]
            nid = parts[1]
            sid = parts[2]
            if((pid,nid,sid) not in list_target):
                list_target.append((pid,nid,sid))

    fn_train = "conll14st-preprocessed"
    fn_test = "conllFile"
    words_list = list(get_dict(fn_train,fn_test))
    words_list = {k: v for v, k in enumerate(words_list)} # convert list to dict
    n = len(words_list)

    A = []
    l = np.zeros(n)
    nid = 0
    pid = 0
    sid = 0

    for line in f:
        parts = line.split()
        if len(parts)>1:
            word = parts[4]
            if(word in words_list):
                index = words_list[word]
                #print(index)
                l[index] = 1
            pid = parts[0]
            nid = parts[1]
            sid = parts[2]
        else:
            if((pid,nid,sid) in list_target):
                A.append(l)
                l = np.zeros(n)
    return A
    