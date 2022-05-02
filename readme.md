# Data linkage & classification 
**google and amazon products** </br>

This project perform a data linkage on two real-world datasets (Part1) and explore different classification algorithms (Part 2).

**Dataset**
datasets for Part 1: Data Linkage
– amazon.csv
– google.csv
– amazon google truth.csv
– amazon small.csv
– google small.csv
– amazon google truth small.csv
datasets for Part 2: Classification:
– life.csv – world.csv

### Part 1 - Data Linkage</br>
Amazon and Google both have product databases. They would like to link the same products in order to perform joint market research.</br>
**Na ̈ıve data linkage without blocking**</br>
For this part, data linkage without blocking is performed on two smaller data sets: amazon small.csv and google small.csv.</br>
**Task - 1A** Using amazon small.csv and google small.csv, implement the linkage between the two data sets.</br>

**Blocking for efficient data linkage**</br>
Blocking is a method to reduce the computational cost for record linkage.</br>
**Task - 1B**: Implement a blocking method for the linkage of the amazon.csv and google.csv data sets.</br>
The product id field corresponds to the idAmazon and idGoogleBase of the amazon.csv and google.csv files respectively. Each row in the output files matches a product to a block.</br>

**Task - 1C**: Report on the Data Linkage project</br>

## Part 2 - Classification</br>
**Comparing Classification Algorithms**</br>
**Task - 2A**: Compare the performance of the following 3 classification algorithms: k-NN (k=5 and k=10) and Decision tree (with a maximum depth of 4) on the provided data. You may use sklearn’s KNeighborsClassifier and DecisionTreeClassifier functions for this task.</br> 
Output is in task2a.csv. </br>
