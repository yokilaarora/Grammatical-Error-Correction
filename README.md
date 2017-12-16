# Grammatical-Error-Correction using Neural Networks

Course Project by  Yokila Arora, Ayush Gupta, Jaspreet Kaur for CS229 (Autumn 2017)

This work focuses on improving the architecture of a Neural Network Global Lexicon Model (NNGLM) and integrate it with a phrase-based Statistical Machine Translation (SMT) model to make an effective system that can automate the task of grammatical error correction with enhanced results. The NNGLM calculates individual word probabilities and uses them to score the correction predictions made by the phrase-based SMT model in response to the input sentences with incorrect grammar. The architecture specifications for the NNGLM have been decided after conducting experiments to study the impact of various network configurations on system performance. Based on those, a Feed-Forward Neural Network (FFNN) with three hidden layers has been used as the NNGLM model. For more details, refer <report link>

File description:
1. utils.py: 
  This file contains utility functions for preprocessing the data for Neural Network model, specifically it returns bag of words representations for the training and test datasets. It also contains a method to save the model parameters. 
  
2. train.py: 
  This file contains the Neural Network architecture implemented in PyTorch. 
  
3. maintest.py: 
  This file contains the code for training and testing the model, along with the hyper-parameters used, and the code for calculation for F0.5 score. 
  
4. create_source.py:
   This file formats the file containing input sentences to a suitable format for the SMT model.
  
5. create_target.py:
   This file formats the file containing output sentences to a suitable format for the SMT model.
  
