# SC1015 Mini-project AY2023

## League of Legends Data Analysis
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)

![image](https://drive.google.com/uc?export=view&id=1xdctuhzj4g0pXHeGj9CBHC-WMw6ghyfM)

### Team members:
- [Oscar](https://github.com/oscarqjh)
- [Dimas](https://github.com/desolaterobot)
- [Mokshit](https://github.com/mokshittjain)

### Project Video Presentation
The video presentation for this project can be found [here](https://youtu.be/-04tsiXo0pM).

---
### Section 1: Dataset and Misc.
#### 1.1 Dataset
All Dataset found in this repository is taken from [Kaggle](https://www.kaggle.com/datasets/chuckephron/leagueoflegends?select=LeagueofLegends.csv). There are multiple files, but we will be mainly using:  
```
├── datasets   
|   ├── LeagueofLegends.csv   
|   ├── Kills.csv   
|   ├── Structures.csv   
```
#### 1.2 Suggested Read Order
We separated our Jupyter notebook into different segments for easier reading. we suggest reading them in the following order:
1. [Initial_EDA.ipynb](https://github.com/oscarqjh/SC1015Project/blob/cf8f089a42c83003e86dbd95d4a29d426aa11a04/Initial_EDA.ipynb)
2. [LogisticRegression.ipynb](https://github.com/oscarqjh/SC1015Project/blob/cf8f089a42c83003e86dbd95d4a29d426aa11a04/LogisticRegression.ipynb)
3. [Champion_Analysis.ipynb](https://github.com/oscarqjh/SC1015Project/blob/cf8f089a42c83003e86dbd95d4a29d426aa11a04/Champions_Analysis.ipynb)
4. [Gold_Analysis.ipynb](https://github.com/oscarqjh/SC1015Project/blob/cf8f089a42c83003e86dbd95d4a29d426aa11a04/Gold_Analysis.ipynb)
5. [TreesAndForests.ipynb](https://github.com/oscarqjh/SC1015Project/blob/cf8f089a42c83003e86dbd95d4a29d426aa11a04/TreesAndForests.ipynb)
6. [KNN.ipynb](https://github.com/oscarqjh/NTU_SC1015_Project/blob/559a04ad22259492f175635d99048c2d8f4cce04/KNN.ipynb)
7. [RNN.ipynb](https://github.com/oscarqjh/SC1015Project/blob/cf8f089a42c83003e86dbd95d4a29d426aa11a04/RNN.ipynb)

#### 1.2 Dependencies
- Matplotlib 3.7
- Pandas 2.0 
- Seaborn 0.12
- Numpy 1.24
- SciPy 1.10.1
- Scikit-learn 1.2.2
- PyTorch 2.0

#### 1.3 Environment Set up
Since we will be using PyTorch's RNN Model, installation of the API is required for `RNN.ipynb`.
> Instructions are taken from https://pytorch.org/get-started/locally/

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

---
### Section 2: Introduction   
#### 2.1 About League of Legends   
League of Legends ([LoL](https://www.leagueoflegends.com/en-sg/)) is a multiplayer online battle arena (MOBA) game developed and published by Riot Games in 2009, with over 153 million monthly active players. The game features two teams of five players, each controlling a champion with unique abilities and strengths. The objective of the game is to destroy the enemy's Nexus while defending their own. An important aspect of the game is objectives management as there are several objectives such as Towers, Dragons, Barons, Kills, Inhibitors, and Gold. Obtaining and managing these objectives well can give a team a significant edge over the enemy.   

#### 2.2 Project Objectives   
In this project, we aim to derive strategies for the game through an in-depth exploratory data analysis(EDA) of the dataset on various variables we deemed important. Furthermore, we also aim to apply these insights that we have obtained through EDA through the usage of various machine learning methods to predict the outcome of any single match.   

#### 2.3 Exploratory Data Analysis
In this section, we tried to explore the dataset and obtain as much insights as possible.    

**Extend of Dataset**   
First, we looked at the the range of competitive matches present in this dataset and found out that it is quite extensive and contains all competitive matches from year 2014 - 2018.   

**Win Rate of Teams**    
Next, we looked at the general win rate of blue vs red team and found out that blue team has a statistically higher chance of winning than red team.   

**Gamelength**   
Next, we analysed the game length of competitive matches and found that mean gamelength is around 37 minutes. Also, we have noticed that the average gamelength of recent years is shorter than earlier years.    

**Objectives**   
Next, we analysed different objectives such Baron kills, Champion kills, Dragon kills, Tower takedown, Inhib takedown and gold difference as possible independant variables to be used in our machine learning models.   

**Champion Kills Analysis**   
Finally, we analysed the Kills variable. We did this through making a kill map and visualised the density of kills happened in the map, and also the time at which the kills happened at. We also looked at the involvement of Jungle Players in the early game (before 15mins) and found out that winning team's jungle shows a different tendency in which lane they choose to gank. Also, we analysed the kills variable as a time series, and confirmed the trend that winning teams' cumulative kill tends to be steeper than that of losing team. This suggest that on an average, winning team is expected to snowball the matches with their early advantages and thus, gaining early advantage should be a important factor to consider. (Maybe it means that playing for ultra late game might not be a good idea?)    

**Champion Usage Analysis**   
We also did a analysis on each champions' use rate and win rate. Through this analysis we were able to obtain valuable information on which champions are *META* from 2014-2018. We can thus derive strategies based on this information, since high usage of a specific champion in such tournment setting would suggest said champion to be considered *strong*, and combining with analysis on the win rate of these champions, we can safely assume that picking these champions with high usage and winrate can boost a team's chance of winning to a certain extent.   

#### 2.4 Hypothesis Testing   
In the previous section we analysed the objectives as possible independant variables for our machine learning model. We first visualised these variables against dependant variable (win/lose) on a Pearson's Correalation Coefficient heat map. From this, we are able to derive the top 5 independant variable that most likely have a effect on the outcome of the game. Finally, we tested our hypothesis using T-test and confirmed our hyposthesis.   

---
### Section 3: Machine Learning Models
In this project, we experimented with a few machine-learning models:   

#### **3.1 Logistic Regression**    
Logistic Regression is a statistical model often used for classification. It estimates the probability of an event (dependent variable) based on a given dataset of independent variables[1].   

For this project, we will be using it to predict the outcome of a game (win/lose) based on "x kills obtained before y minutes". For the model based on "x kills obtained before 5 minutes" we are able to obtain a model with **~64%** accuracy.

We also modelled it on tower takedown and tried predicting the outcome of a game based on "x tower takedowns before y minutes". Despite being able to achieve a much better accuracy of **~71%**, we observed a weird phenomenon where getting 1-3 tower takedowns before y minutes actually lowers the probability of the team to win. The probability of winning only becomes more than 50% after getting 4 tower takedowns or more before y minutes.   

#### **3.2 Single/Multi-Variated Decision Tree**   
A decision tree is a non-parametric supervised learning algorithm utilised for classification and regression tasks. Multi Variated Decision Tree models are a type of classification model that is based on multiple variables[2]. In this project, we used both single and multi-variated decision tree on different analysis.

For this project we used single-variated decision tree on 'final gold difference' as variable to predict a match's outcome. Despite obtaining an accuracy of **~95%** we deemed it to be not a good model due to some flaws that it possess. This model requires complete information up until the end of a match to be able to predict a match's outcome. This makes it not practical since such a model will be useless in a real life setting. However, a possible improvement we can make is to use only gold difference up until x minute for classification. Though, there will also be shortcoming for this approach due to its nature that gold difference can vary wildly from every single minute. Hence, we will try to seek for better models for prediction of a game's outcome.    

Following this, we also used multi-variated decision tree on *first tower timing, first inhib timing, first dragon timing, first kill timing and first baron timing*. We were able to obtain an accuracy of **~92%** however there are many flaws in this model which we will provide more details on in the Jupyternb itself.   

#### **3.3 Random Forest**    
Random Forest is a commonly used machine learning algorithm which combines the output of multiple decision trees to reach a single result[3].    

For this project, we first used normal classification trees with depth of 5 on *first tower timing, first inhib timing, first dragon timing, first kill timing and first baron timing*. And combined them using the random forest model of 400 trees and was able to obtain a pretty high classification accuracy of **~93%**. Yet, there are still some flaws in this model since it also runs the risk of overfitting, and the inability to predict the match's outcome if there are presence of missing values.

#### **3.4 K-Nearest Neighbour**

The KNN model is a supervised learning classifier which uses proximity to make classifications or predictions about the grouping of individual data points. The basic idea is that similar points can be found near one another and thus classified base of *majority vote*[4].   

In this project, we used counts of events occurred at x minutes into the game. We first used the full game length for our initial model, which produced a very high classification accuracy of **~96%**, however we were unsatisfied with this model as it has little practical value since it it using the full game's data. Meaning that it could only predict the final outcome of the game after the match is over. So next, we decided to only use partial game data - data until 30 minutes of the game. For this model, we were able to obtain a accuracy of **~81.4%**, a pretty satifactory result.   

#### **3.5 Recurrent Neural Network**   
RNN is a class of artificial neural networks which uses sequential data. A characteristic feature of RNNs is that they are about to take a hidden output from the previous iteration as inputs for the next iteration[5]. 

For this project, we used sequential data of "difference between events that occurred at every minute from 0 to x minutes" for different variables which we identified to be important such as Baron kills, Dragon kills, Tower takedowns, Gold difference and Champion kills to predict the outcome of the game. With PyTorch's RNN model[6] we are able to obtain an accuracy of **~84%**, the best so far for partial data prediction within our project. Overall, this is the model we are most satisfied with.   

---
### Section 4: Conclusion   
#### 4.1 Possible Strategies Derived   
**Early Jungle Invasion**   
From our analysis on kills in `LogisticRegression.ipynb`, we can see that getting 1-2 kills before 3 minutes translates to a significant increased chance of winning however, anything more only yields marginal increase in chances of winning, this sheds some light into common strategies such as early jungle invasion. Early jungle Invasion is a viable strategy to potentailly get some very early kills within the first 1-2 minutes of the game. However, after obtaining the first 2 kills, subsequent kills yields *marginal* advantage for the team. Hence, it might be wiser to not greed for more kills after getting the first kill, but instead focus on laning phase, where the effects of a kill is much more significant.   

**Playing for Early Dominance**   
From our analysis on cumulative kills in `Initial_EDA.ipynb`, we can see that the line for cumulative kills of winning team being steeper than the losing team. This suggests that there is a tendency for early advantages to snowball into the lategame. Meaning that it might be a viable and a wiser strategy to choose a team composition that can secure early game advantage instead of choosing a late game team composition that tries to survive long enough for their champion to peek in performance.   

**Jungle Player Gank**   
Although the observation from our EDA might not show the causal and effect relation between Jungle involvement and chance of winning. It might be wise for jungles to focus on all lanes instead of just the bot lane as statistics in our kill map in `Initial_EDA.ipynb` shown that winning teams generally have the jungle's involvement more spread out between all 3 lanes, while loosing teams jungle involvement then to focus on the bottom lane.   

**Choosing Team**   
Although in competitive matches teams cant choose which side (blue or red) to start from. However, from our analysis, blue team has a statistically higher rate of winning. So, whenever possible, players should always choose to be on the blue team.   

**Choosing Champions**   
If we disregard other factors such as team synergy and counter picks, picking the same champions with the highest usage rate and win rate according to our analysis in `Champion_Analysis.ipynb` should yield a team the best chance to win.    

#### 4.2 Prediction of Match Outcome   
We have experimented with multiple machine learning models. We started of with uni-variated model (Logistic Regression) which yield an accuracy of **~64%** for *cumulative kills* and an accuracy of **~71%** for *cumulative towers*. This is a very basic model which can only incorporate 1 variable at a time.      

Next, we used what we have learned in this course (Uni-Variated Decision Tree) and obtained an model with accuracy of **~95%**. However, we deemed this model not to be so good since it runs the risk of overfitting, moreover, it requires complete information on golddiff at the end of the match to classify the outcome of a match. The notable flaws in this model are that gold diff can varies greatly every minutes, hence result is only reliable on the final gold diff, and also, it is not very practical since only being able to classify the outcome of a match after it ended is not useful.   

Next, we used the Random Forest model on *first tower timing, first inhib timing, first dragon timing, first kill timing and first baron timing* and was able to obtain an accuracy of **~93%**. Similar to the decision tree, there are many flaws with this model which we are unsatisfied with despite its high accuracy.    

Then, we came across the KNN classifier. Using this model, for the first time in this project were we able to predict the outcome of a match with only partial game data - until 30 minutes into the game. Furthermore, the accuracy is pretty decent, at **~81.4%**. 

Lastly, we decided to use the RNN model with time series input of *kill diff, tower diff, baron diff, dragon diff and gold diff* before 30 minutes. We were able to obtain a model with satisfactory accuracy of **~84%**. Overall, this is the model we are most satisfied with due to its ability to predict match outcomes with based on multiple variables and also is based on partial game data.   

#### 4.3 Ending Notes   
We are very pleased with our results, however we are also aware of the difficulties to truly predict a match's outcome with extremely high accuracy and precision due to the multitude of factors which can affect a match's outcome, especially those factors which is hard to be expressed in the form of tangible data such as a player's skill level, playstyle, condition. Moreover, there are also factors such as champion synergy and champion counter picking which are important but unstructured datas which can be hard to model after. Moreover, we did not fully utilise all the other files available in this dataset, such as `monsters.csv` and `bans.csv` due to time constraints. I believe that more insights and better models can be derived if we are able to incoporate all the available datas.       

---

*Submission: This repository is submitted to Nanyang Technological University Singapore as a project for module SC1015 AY2023*

<h3 align="center">Reference</h3>

[1]: [*What is Logistic Regression*. IBM, 2023](https://www.ibm.com/topics/logistic-regression)    
[2]: [*What is Decision Tree*. IBM, 2023](https://www.ibm.com/topics/decision-trees)   
[3]: [*What is Random Forest*. IBM, 2023](https://www.ibm.com/topics/random-forest)      
[4]: [*What is KNN*. IBM, 2023](https://www.ibm.com/topics/knn)
[5]: [*What is RNN*. IBM, 2023](https://www.ibm.com/topics/recurrent-neural-networks)    
[6]: [PyTorch's RNN model](https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/)    


