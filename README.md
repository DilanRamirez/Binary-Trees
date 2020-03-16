# Binary-Trees

## Description
The file magic04.txt contains data from a gamma ray detection experiment. Each line in the file represents an observation, with the first ten items, all floating point numbers, describing the data collected by the detector, and the last item, a character (g or h), indicating whether the detection corresponds to a gamma ray (g) or not (h). The task for this lab is to construct a decision tree, as described in class, for this problem. The program decisiontree.py provides code that reads the data, splits it into training and testing parts, and implements a one-node decision tree from the training data to classify the test data. For this lab, the decision tree functions will be extended to build trees with more than one node and use these trees to classify the data, hopefully improving accuracy.
 * a) Extend the BuildDT function. In the current implementation, leftChild and rightChild are classification labels (0 or 1), in the extended implementation, they should be references to decision trees, provided the dataset is large enough and the goal accuracy has not been attained.
 * b) Experiment with different values of parameters to obtain the highest possible accuracy on     the test set.
 * c)	Display the following statistics about the tree generated: number of nodes, number of times each attribute is used for splitting, and tree height.
 * d)	Generate multiple trees from the training data and average the results to make predictions. Determine if accuracy can be improved doing this
 
 ## Author 
 Dilan R. Ramirez
