# SC1015 Mini-project AY2023

## League of Legends Data Analysis

### Team members:
- Oscar
- Dimas
- Mokshitt

### Section 1: Dataset and Misc.
#### 1.1 Dataset
All Dataset found in this repository is taken from [Kaggle](https://www.kaggle.com/datasets/chuckephron/leagueoflegends?select=LeagueofLegends.csv). There are many datasets, but we will be mainly using:  
```
├── datasets   
|   ├── LeagueofLegends.csv   
|   ├── Kills.csv   
```
#### 1.2 Suggested Read Order
We seperated our jupyter notebook into different segments for easier reading, we suggest reading in the following order:
1. EDA.ipynb
2. LogisticRegression.ipynb
3. RNN.ipynb

#### 1.2 Dependancies
- Matplotlib 3.7
- Pandas 2.0
- Seaborn 0.12
- Numpy 1.24
- SciPy 1.10.1
- Scikit-learn 1.2.2
- PyTorch 2.0

#### 1.3 Environment Set up
Since we will be using PyTorch's RNN Model, installation of the API is required for `RNN.ipynb`
> Instructions taken from https://pytorch.org/get-started/locally/

#### Without Anaconda (skip this if using Anaconda)
Create a Conda environment:
```
conda create -n env_pytorch python=3.7
```
Activate Conda environment:
```
conda activate env_pytorch
```
#### With Anaconda (important)
Install PyTorch using pip:
```
pip install torchvision
```


### Section 2: Introduction

### Section 3: Machine Learning Models
In this project, we experimented with a few machine learning models:   

#### 3.1 Logistic Regression    
Logistic Regression is a statistical model often used for classfication. It estimates the probability of an event (dependant variable) based on a given dataset of independent variables[1].   

For this project, will be using it to predict the outcome of a game (win/lose) based on "x kills obtained before y minutes". For the model based on "x kills obtained before 5 minutes" we are able to obtain a model with ~64% accuracy

#### 3.2 Multi Variated Decision Tree   

#### 3.3 Random Forest    
Random Forest is a commonly used machine learning algorithm which combines the output of multiple decision trees to reach a single result[3].

#### 3.4 Recurrent Neural Network   
RNN is a class of artificial neural networks which uses sequential data. A characteristic feature of RNN is that they are about to take a hidden output from previous iteration as inputs for the next iteration[4]. 

For this project, we used sequential data of "difference between events occurred at every minute from 0 to x minutes" for different variables which we identified to be important such as Baron kills, Dragon kills, Tower takedowns, Gold difference and Champion kills to predict the outcome of the game. With the PyTorch's RNN model[5] we are able to obtain an accuracy of ~84%, the best so far within our project.

### Section 4: Conclusion   
#### 4.1 Possible Strategies Derived   

#### 4.2 Final Comments   


<h3 align="center">Reference</h3>
<a id="1">[1]</a> 
Dijkstra, E. W. (1968). 
Go to statement considered harmful. 
Communications of the ACM, 11(3), 147-148.



[1]: https://www.ibm.com/topics/logistic-regression#:~:text=Resources-,What%20is%20logistic%20regression%3F,given%20dataset%20of%20independent%20variables.    
[3]: https://www.ibm.com/topics/random-forest#:~:text=Random%20forest%20is%20a%20commonly,both%20classification%20and%20regression%20problems.   
[4]: https://www.ibm.com/topics/recurrent-neural-networks    
[5]: https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/    
[6]: https://realpython.com/logistic-regression-python/#single-variate-logistic-regression

