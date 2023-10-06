# Math-Practical


Copy of Practice_Questions.ipynb_
Notebook unstarred
Practical Test - I, Spring 2023
CSM301 : Mathematics for Programming III
B.Sc Computer Science(AI & DS)
Year II, Semester II
Time: 2 Hours
Max. Marks: 50

Rename the file with your Enrollment number on the top immediately on receipt of this question file. All questions are compulsory, and marks are given at the end of each question. Parts of a question should be answered together. Please use Python programming to provide responses to the given questions.
Question 1

Study the data presented below and answer the following questions:

StaffAgeSalary12515,00022618,00032516,00042314,00053015,00062915,00072312,00083417,00094020,000103016,000115160,000

    Create a dataframe using the information provided in the table above. [2]

    Compute the harmonic mean, variance and mode for the columns labeled 'Age' and 'Salary'. [3]

    Create a bar chart to illustrate the salary data of the staff, and label the axis with the proper column names. [2]

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


Double-click (or enter) to edit

#Coverting into dataframe

data_frame = {'Staff':[1,2,3,4,5,6,7,8,9,10,11], 'Age':[25,26,25,23,30,29,23,34,40,30,51],'Salary':[15000,18000,16000,14000,15000,15000,12000,17000,20000,16000,60000]}

df = pd.DataFrame(data_frame)
df

#Question 2

#Harmonic Mean
from scipy import stats
print("The mean of age :", stats.hmean(df['Age'])," Salary :", stats.hmean(df['Salary']) )

#Mode

print("The mean of age :", stats.mode(df['Age'])," Salary :", stats.mode(df['Salary']) )

#Variance
print("The variance of Age :",df['Age'].var(), " Salary :", df['Salary'].var())
# print("The variance of age:", stat.)

The mean of age : 28.902714234790153  Salary : 16646.296230880638
The mean of age : ModeResult(mode=23, count=2)  Salary : ModeResult(mode=15000, count=3)
The variance of Age : 71.87272727272727  Salary : 181963636.3636363

#Question3 Plotting Barchart

sns.barplot(x = df['Staff'], y = df['Salary'])
plt.title("Salary Data of Staff")

plt.show()

Question 2
↳ 7 cells hidden
Question 3
↳ 6 cells hidden
Question 4

You have three bags that each contain 100 marbles:

    Bag 1 has 75 red and 25 blue marbles;
    Bag 2 has 60 red and 40 blue marbles;
    Bag 3 has 45 red and 55 blue marbles.

You choose one of the bags at random and then pick a marble from the chosen bag, also at random. What is the probability that the chosen marble is blue? [4]

# P_A be choosing one of the bag at random
#P_AB be picking a marble from choose bag
#P_B be chosen marble is blue
#For bag 1
P_A = 1/3
P_B = 25/100
#P_AB = P_AnB/P_B
P_AB = (1/3*25/100)/(25/100)
print("Bag1: ",P_AB)

 # For bag2
P_A1 = 1/3
P_B1 = 40/100
#P_AB = P_AnB/P_B
P_AB1 = ((1/3)*(40/100))/(40/100)
print("Bag2: ", P_AB1)

 # For bag3
P_A2 = 1/3
P_B2 = 55/100
#P_AB = P_AnB/P_B
P_AB2 = (1/3*55/100)/(55/100)
print("Bag3: ", P_AB2)

print ("The probability that the chosen marble is blue: ", P_AB*P_AB1*P_AB2)

Bag1:  0.33333333333333326
Bag2:  0.3333333333333333
Bag3:  0.33333333333333326
The probability that the chosen marble is blue:  0.037037037037037014

Question 5

A coin is either fair or biased, with a 75% chance of being biased. The biased coin comes up heads with a probability of 0.6, while the fair coin comes up heads with a probability of 0.5. If we toss the coin and it comes up heads, what is the probability that it is the biased coin? [4]

P_A = ((0.25*0.5)/0.6) + ((0.75*0.5)/0.5)
P_A
print("The probability that it is the biased coin: ", P_A)

The probability that it is the biased coin:  0.9583333333333334

Question 6

Assume there are six multiple choice questions in a brief quiz. Each question contains four possible responses, any of which is correct. On every question, a student makes a guess.

    Find the probability that student gets exactly 4 questions right. [3]

    Find the mean, variance, and standard deviation of the distribution. [3]

from scipy.stats import binom
#Question1
# p = 1/4 = 0.25 probability of getting correct answer

p = binom.pmf(n = 6, k = 4, p = 0.25)

print("The probability that student gets exactly 4 questions right: ", p , 'or ', 3.3, "%")

The probability that student gets exactly 4 questions right:  0.03295898437499997 or  3.3 %

#Question2
print( "The mean of the distribution: ", binom.mean(n = 6, p = 0.25))
print()
print("The variance of the distribution: ", binom.var(n = 6, p = 0.25))
print()
print("The standard deviation of the distribution: ", binom.std(n = 6, p = 0.25))

The mean of the distribution:  1.5

The variance of the distribution:  1.125

The standard deviation of the distribution:  1.0606601717798212

Question 7

The continuous random variable X has probability density function f(x), given by

f(x)={29(5−x),0,2≤x≤5otherwise

Find the E(X2),Var(X), and standard deviation of X. [1.5 + 2 + 0.5]

import sympy as smp

x = smp.Symbol('x')

fx = (2/9)*(5-x)
px = smp.integrate(fx,(x,2,5))
# print(px)

fx1 = x* (2/9)*(5-x)
EX = smp.integrate(fx1, (x, 2,5))
EX

fx2 = (x**2)*(2/9)*(5-x)
EX2 = smp.integrate(fx2,(x,2,5))
print("𝐸(𝑋2): ", EX2 )
print()

#Variance
var = EX2 - (EX)**2

print("Variance of X: ", var)
print()

#standard deviation
sd = (var)**(0.5)
print("Standard deviation of X: ", sd)

𝐸(𝑋2):  9.50000000000000

Variance of X:  0.500000000000002

Standard deviation of X:  0.707106781186549

Question 8

Load the built-in dataset 'geyser', which is available in seaborn library and answer the following questions:

    Determine the correlation coefficient for the columns 'duration' and 'waiting'. [1.5]

    Visualize the correlation between the two colmuns using heatmap and display the correlation score. [1.5]

    Calculate the covariance for every possible pair of columns within a dataset. [2]

gs = sns.load_dataset('geyser')
gs.head()

gdf = gs.select_dtypes(include=(['int', 'float']))
gdf.head()

#Question1

gdf.corr(method = 'pearson')

#question2

sns.heatmap(gdf.corr(method = 'pearson'), annot=True, cmap='coolwarm')

#Question3
gs.cov()

Question 9

A company produces boxes of cereal with a weight that follows a uniform distribution between 10 ounces and 12 ounces. What is the probability that a randomly selected box of cereal weighs between 11 ounces and 11.5 ounces? [3]

from scipy.stats import uniform

pu = uniform.cdf(x = 11.5, loc = 10, scale = 2)-uniform.cdf(x=11, loc=10, scale = 2)
pu
print("The probability that a randomly selected box of cereal weighs between  11 ounces and  11.5 ounces: ", pu)

The probability that a randomly selected box of cereal weighs between  11 ounces and  11.5 ounces:  0.25

Question 10

Import the 'taxis' dataset, which is a built-in dataset within the seaborn library, and then standardize the 'distance' column using standard normal distribution. Then, consider a scenario where an unknown individual has boarded the taxi, and determine the likelihood of this person traveling a distance greater than 25 kilometers. [5]

taxis = sns.load_dataset('taxis')
taxis.head()

taxis['distance'].max()

36.7


taxis['distance'].min()

0.0

mean = taxis['distance'].mean()
print("Mean: ",mean)

sd = taxis['distance'].std()
print("Standard deviation: ",sd)

Mean:  3.0246168195243133
Standard deviation:  3.8278670010117537

#Normalized

normalized = (taxis['distance']-mean)/(sd)
normalized.head()

0   -0.372170
1   -0.583776
2   -0.432256
3    1.221407
4   -0.225874
Name: distance, dtype: float64

z1 = (25-mean)/sd
z2 = (36.7-mean)/sd

from scipy.stats import norm
prob = norm.cdf(z2) - norm.cdf(z1)
prob

4.708868961422752e-09

The probability of the person traveling a distance greater than  25 kilometers:  4.708868961422752e-09





Unit one notes

x = [1,3,4,1.5,6,2,8]
y = [-2,4,3,6,1,1,10]

import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.title('Scatter Plot') #To format the chart title
plt.xlabel('x')           #To format the x-axis title
plt.ylabel('y')           #To format the y-axis title
plt.show()

sizes = [10, 40, 60, 80, 100,50,70]
colors = ['r', 'b', 'y', 'g', 'k','b','b']
plt.scatter(x, y, s=sizes, c=colors)
plt.show()

import numpy as np
x = np.linspace(0, 10, 10000)
y = np.cos(x)

plt.plot(x, y)
plt.show()

labels = ['Type 1', 'Type 2', 'Type 3']
counts = [2, 3, 5]

plt.bar(labels, counts)
plt.show()



rom matplotlib import pyplot as plt
import numpy as np

# Creating dataset
ds = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])

# Creating histogram
fig, ax = plt.subplots(figsize =(8, 5))
ax.hist(ds, bins = [0, 25, 50, 75, 100])

# Show plot
plt.show()



# Import libraries
from matplotlib import pyplot as plt
import numpy as np

# Creating dataset
cars = ['Kia', 'Van', 'EV', 'TESLA', 'Alto', 'Swift']

data = [23, 17, 35, 2, 30, 23]

# Creating plot
fig = plt.figure(figsize =(9, 6))
plt.pie(data, labels = cars,autopct='%1.1f%%', explode=(0,0,0,0.2,0,0))

# show plot
plt.show()


import numpy as np
import matplotlib.pyplot as plt

my_map = np.random.randn(6, 6)

plt.imshow(my_map)
plt.colorbar()
plt.show()



# Calculate arithmetic mean
import statistics as stat

print('The mean of given data is: ', stat.mean([1,9,5,6,4,7,3,6]))

# Create dataframe

import pandas as pd
import numpy as np

#Create a DataFrame
d = {
    'Name':['Pema','Dechen','Wangmo','Tshewang','Karma','Yangden','Sonam'],
   'Math_Score':[62,47,55,74,31,77,85],
   'AI_Score':[89,87,67,55,47,72,76]}

df = pd.DataFrame(d)
df


# mean of the dataframe
df['AI_Score'].mean()


df1 = df[['AI_Score', 'Math_Score']] # Mean of two columns
df1.mean()

# mean of the specific column
df['Math_Score'].mean()

# Calculate geometric mean
from scipy import stats

print(stats.gmean([4,11,15,16,5,7]))

import pandas as pd
import numpy as np
from scipy import stats

#Create a DataFrame
d = {'Name':['Dorji','Sonam','Chador','Shyam','Zangmo','Om','Sushmita'],
   'Stats_Score':[62,47,55,74,31,77,85],
   'Algebra_Score':[89,87,67,55,47,72,76]}

df = pd.DataFrame(d)
df

# Geometric Mean of the column in dataframe
from scipy import stats

stats.gmean(df.iloc[:,1:3],axis=0)

# Row wise geometric mean of the dataframe
from scipy import stats

stats.gmean(df.iloc[:,1:3],axis=1)

# Geometric mean of the specific column
stats.gmean(df.loc[:,"Stats_Score"])


# calculate harmonic mean
from scipy import stats

print(stats.hmean([4,11,15,16,5,7]))


import pandas as pd
import numpy as np
from scipy import stats

#Create a DataFrame
d = {'Name':['Dorji','Sonam','Chador','Shyam','Zangmo','Om','Sushmita'],
   'Stats_Score':[62,47,55,74,31,77,85],
   'Algebra_Score':[89,87,67,55,47,72,76]}

df = pd.DataFrame(d)
df


# Harmonic Mean of the  column in dataframe
from scipy import stats

stats.hmean(df.iloc[:,1:3],axis=0)


# Row wise harmonic mean of the dataframe
from scipy import stats

stats.hmean(df.iloc[:,1:3],axis=1)

# Row wise harmonic mean of the dataframe
from scipy import stats

stats.hmean(df.iloc[:,1:3],axis=1)



# Row wise harmonic mean of the dataframe
from scipy import stats

stats.hmean(df.iloc[:,1:3],axis=1)


# median of the dataframe
df.iloc[:,1:4].median()


# median of the specific column
df.loc[:,"ML_Score"].median()


# calculate mode or most repeated value
import statistics as stats

stats.mode(['lion', 'cat', 'cat','dog','tiger'])


# Mode of the dataframe (or column mode of the dataframe)
df.mode()


df.corr(method='pearson') #Pairwise correlation of all columns in the dataframe

import seaborn as sns
sns.heatmap(df.corr(method='pearson'), annot = True)



Unit 2

A = {1,2,3,7,8}
B = {3,7,9}

print(1 in A) # to check the elements in the set

# Now define the universal set with the help of arange() function
# Now define the universal set with the help of arange() function
universal = set(np.arange(10))

# Type of universal variable above
type(universal)

A = set(np.arange(2,10,2))
B = set(np.arange(1,9,3))

# A union B can be calculated by the function union()
A.union(B)

# A intersection B can be calculated by the function intersection()
A.intersection(B)

# Difference
A.difference(B)

# A_Compliment can be calculated using the difference() function
A_Compliment = universal.difference(A)
A_Compliment



/*


Example 9: Two unbiased dice are thrown once and the total score is observed. Use a simulation to find the estimated probability that the total score is even or greater than 7?

Solution: Following are the steps to be followed;

    Run the experiment 1000 times (roll 2 dice 1000 times, and sum the result)
    Keep track of the number of times that the sum was either greater than 7 or even
    Divide the number from step 2 by the number of iterations (1000)

Awesome! Now let's code this and see what the probability is!


import numpy as np
import random


#Function for roll the Dice

def roll_the_dice(n_simulations = 1000):
  count = 0

  #Each iteration of the for loop is trial
  for i in range(n_simulations):

    #Roll each Die
    die1 = random.randint(1,7)
    die2 = random.randint(1,7)

    #Sum the values to get the score
    score = die1 + die2

    #decide if we should add it to the count
    if score % 2 == 0 or score > 7:
      count += 1
  return count/n_simulations

string = 'The probability of rolling an even number or greater than 7 is:'
print(string, np.round(roll_the_dice()*100, 2), '%')


*/

/*



Example 10: A box contains 10 white balls, 20 reds and 30 greens. Draw 5 balls with replacement. what is the probability that:

A. 3 white or 2 red.

B. All 5 are the same color.

Solution: We will pick our balls during each round and count the number of times. Below is code to do that.

import numpy as np
import random
 # Let's set up the dictionary that we will use for this question
 # This dictionary will allow us to randomly choose a color
d = {}
for i in range(61):
  if i < 10:
    d[i] = 'white'
  elif i > 9 and i < 30:
    d[i] = 'red'
  else:
    d[i] = 'green'

#Initialize important variables
n_simulations = 10000
part_A_total = 0
part_B_total = 0

for i in range(n_simulations):

  #make a list of the colors that we choose
  list = []
  for i in range(5):
    list.append(d[random.randint(0,59)])

  #convert it to a numpy
  list = np.array(list)

  #find the number of each that we picked
  white = sum(list == 'white')
  red = sum(list == 'red')
  green = sum(list == 'green')

  #Keep track if the combination met the above critria
  if white == 3 and red == 2:
    part_A_total += 1

  if red == 5 or white == 5 or green == 5:
    part_B_total +=1

print('The probability of 3 white and 2 red is: ', part_A_total/n_simulations*100, '%')
print('The probability of all the same color is: ', part_B_total/n_simulations*100, '%')
*/


# A Python program to print all
# permutations using library function
from itertools import permutations

# Get all permutations of [1, 2, 3]
perm = permutations([1, 2, 3])

# Print the obtained permutations
for i in tuple(perm):
  print(i)


# A Python program to print all
# permutations of given length
from itertools import permutations

# Get all permutations of length 2
perm = permutations([1,2,3], 2)

# Print the obtained permutations
for i in tuple(perm):
	print (i)


# A Python program to print all
# combinations of given length
from itertools import combinations

# Get all combinations of [1, 2, 3]
# and length 2
comb = combinations([1, 2, 3], 2)

# Print the obtained combinations
for i in tuple(comb):
	print(i)


# A Python program to print all
# combinations of a given length
from itertools import combinations

# Get all combinations of [1, 2, 3]
# and length 2
comb = combinations([1,2,3], 2)

# Print the obtained combinations
for i in tuple(comb):
	print (i)


# A Python program to print all combinations
# of given length with unsorted input.
from itertools import combinations

# Get all combinations of [2, 1, 3]
# and length 2
comb = combinations([2, 1, 3], 2)

# Print the obtained combinations
for i in tuple(comb):
	print (i)


# A Python program to print all combinations
# with an element-to-itself combination is
# also included
from itertools import combinations_with_replacement

# Get all combinations of [1, 2, 3] and length 2
comb = combinations_with_replacement([1, 2, 3], 2)

# Print the obtained combinations
for i in tuple(comb):
	print (i)



/*


We’re going to calculate the probability a student gets an grade A (80% and above) in math, given that they miss 10 or more classes.

import numpy as np
import pandas as pd
df = pd.read_csv('student-mat.csv')
df.head()

len(df)

df.shape

Add a boolean column called grade_A noting if a student achieved 80% or higher as a final score. Original values are on a 0–20 scale so we multiply by 5.
df['grade_A'] = np.where(df['G3']*5 >= 80, 1, 0)


Make another boolean column called high_absenses with a value of 1 if a student missed 10 or more classes.
df['high_absenses'] = np.where(df['absences'] >= 10, 1, 0)
df.head()

df['count'] = 1
df.head()

df = df[['grade_A','high_absenses','count']]
df.head()


pd.pivot_table(df, values='count', index=['grade_A'], columns=['high_absenses'],
               aggfunc=np.size, fill_value=0)

 */




Unit 3


#Visualize using bar plot
fig = plt.figure(figsize = (5,3))
plt.bar(data.x, data['P(X = x)'], color = 'maroon')
plt.title('Probability Distribution')
plt.xlabel('x')
plt.ylabel('P(X = x)')
plt.show()


The code to visualize the bernoulli distribution for different value of p is given below. As a example, lets take p=0.7.

import matplotlib.pyplot as plt
from scipy.stats import bernoulli
# Instance of Bernoulli distribution with parameter p = 0.7
bd = bernoulli(0.7)

# Outcome of experiment can take value as 0, 1
X = [0, 1]

# Create a bar plot; Note the usage of "pmf" function to determine the
#probability of different values of random variable
plt.figure(figsize=(5,3))
plt.bar(X, bd.pmf(X), color='green')
plt.title('Bernoulli Distribution (p=0.7)', fontsize='12')
plt.xlabel('Values of Random Variable X (0, 1)', fontsize='12')
plt.ylabel('Probability', fontsize='12')
plt.show()


/*
An exciting computer game is released. Sixty percent of players complete all the levels. Thirty percent of them will then buy an advanced version of the game. Among 15 users, what is the expected number of people who will buy the advanced version? What is the probability that at least two people will buy it?

from scipy.stats import binom

#Calculate first expected value or mean
E_X = binom.moment(1, 15, 0.18)
print('Expectation of X is:', round(E_X,2))

#calculate binomial probability
result_0 = binom.pmf(k=0, n=15, p=0.18)
result_1 = binom.pmf(k=1, n=15, p=0.18)

#Print the result
print("Binomial Probability when X = 0: ", round(result_0, 2),
      '\nBinomial Probability when X = 1:', round(result_1, 2))

The required probability that is P(X≥2)=1−P(0)−P(1)

required_prob = 1 - result_0 - result_1
print('The required probability: ', round(required_prob, 2))
/*


#Setting the values of n and p
n, p = 15, 0.18

#Defining the list of x values
x = list(range(n+1))

#Create DataFrame which consist of x, pmf, and cdf
rv = binom(n, p)
df = pd.DataFrame({'x': x, 'pmfs': rv.pmf(x), 'cdfs': rv.cdf(x)})
df.head()

plt.figure(figsize = (5,3))
plt.bar(x, df.pmfs, color = 'green')
plt.xlabel('x')
plt.ylabel('Probabilities')
plt.title('Probability Distribution')
plt.show()


plt.figure(figsize = (5,3))
plt.plot(x, df.cdfs, color = 'maroon')
plt.xlabel('x')
plt.ylabel('cdfs')
plt.title('Cumulative Distribution')
plt.show()

mean = binom.mean(n = 15, p = 0.18)
var = binom.var(n = 15, p = 0.18)
std = binom.std(n = 15, p = 0.18)

print('\nMean: ', round(mean, 2), '\nVariance: ', round(var, 2),
      '\nStandard deviation: ', round(std, 2))


from scipy.stats import geom
import matplotlib.pyplot as plt

p_6 = geom.pmf(k = 6, p = 0.04) #Mass function of geometric distribution
print('Prob. of first defective:', round(p_6, 3))

from scipy.stats import geom
import matplotlib.pyplot as plt
import pandas as pd

# X = Discrete random variable representing number of throws
# p = Probability of the perfect throw
#Create a DataFrame which consist of x, pmfs, and cdfs
x = list(range(1,11))
p = 0.6
df = pd.DataFrame({'x': x, 'pmfs':geom.pmf(x, p), 'cdfs': geom.cdf(x, p)})
df.head()

fig = plt.subplots(figsize=(5, 3))
plt.bar(x, df.pmfs, color = 'green')
plt.ylabel("Probability")
plt.xlabel("X - No. of Throws")
plt.title("No. of Throws Vs Probability")
plt.show()

fig = plt.subplots(figsize=(5, 3))
plt.plot(x, df.cdfs, color = 'maroon')
plt.ylabel("cdf")
plt.xlabel("X - No. of Throws")
plt.title("No. of Throws Vs cdfs")
plt.show()

mean = geom.mean(p = 0.6)
var = geom.var(p = 0.6)
std = geom.std(p = 0.6)

print('\nMean: ', round(mean,2), '\nVariance: ', round(var,2),
      '\nStandard deviation: ', round(std,2))


from scipy.stats import poisson

#Generate random values from Poisson distribution with mean=3 and sample size=10
poisson.rvs(mu = 3, size = 10)


from scipy.stats import poisson

#calculate probability
prob = poisson.pmf(k=5, mu=3)
print('Required prob.: ', round(prob, 4))


from scipy.stats import poisson

#calculate probability
prob = poisson.cdf(k=4, mu=7)
print('Required prob.: ', round(prob, 4))


from scipy.stats import poisson

#calculate probability
prob = 1 - poisson.cdf(k=20, mu=15)
print('Required prob.: ', round(prob, 4))


from scipy.stats import poisson
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


x = np.linspace(1,10,10) # Values X takes

#Create DataFrame which consist of x, pmfs, and pdfs
df = pd.DataFrame({'x': x, 'pmfs':poisson.pmf(x, mu = 7),
                   'cdfs': poisson.cdf(x, mu = 7)})
df.head()


fig = plt.figure(figsize = (5,3))
plt.bar(x, df.pmfs, color = 'maroon') # Visualize using bar chart
plt.xlabel('X = x')
plt.ylabel('Prob. mass functions')
plt.title('Poisson Distribution')
plt.show()

fig = plt.figure(figsize = (5,3))
plt.plot(x, df.cdfs, color = 'green') # Visualize using line plot
plt.xlabel('X = x')
plt.ylabel('cdf')
plt.title('cdfs vs. x values')
plt.show()



mean = poisson.mean(mu = 7)
var = poisson.var(mu = 7)
std = poisson.std(mu = 7)

print('\nMean: ', round(mean,2), '\nVariance: ', round(var,2),
      '\nStandard deviation: ', round(std,2))


#Using Sympy library
import sympy as smp

x = smp.Symbol('x')

p_x = smp.integrate(x, (x, 0, 0.5))

print('The probability is: ', round(p_x,3))



/*



Example 2: A random variable X has density
f(x)=3x−4,x≥1

    Check whether f(x) is density function.
    If f(x) is density function, then find E(X), Var(X), and Standard deviation (σ).

	
#Solution of part 1, to check f(x) is pdf, integral of f(x) should be equal to 1.
import sympy as smp

x = smp.Symbol('x')
fx = 3*x**(-4)

smp.integrate(fx, (x, 1, smp.oo))

#It is pdf since integral over the limit is 1.

#Solution of part 2, E(X)
import sympy as smp

x = smp.Symbol('x')
fx1 = x*3*x**(-4)
E_X = smp.integrate(fx1, (x, 1, smp.oo))
print('Expectation of X is:', E_X)

#Var(X) = E(X^2) - (E(X))^2
fx2 = (x**2)*(3)*(x**(-4))
E_X2 = smp.integrate(fx2, (x, 1, smp.oo))
print('Expectation of X^2 is:', E_X2)

Var_X = E_X2 - (E_X)**2
print('Variance of X is: ', Var_X)

#Standard deviation

std_X = (Var_X)**(0.5)
print('The standard deviation of X is: ', std_X)	

*/


/*
 If X is uniformly distributed over (0,10), calculate the probability that a) X<3, b) X>7, c) 1<X<6.
	
#P(X < 3)
from scipy.stats import uniform
prob = uniform(loc = 0, scale = 10).cdf(3) - uniform(loc = 0, scale = 10).cdf(0)
print('P(X < 3):', prob)

#P(X > 7)
prob = uniform.cdf(x = 10, loc = 0,  scale = 10) - uniform.cdf(x = 7, loc = 0,
                                                               scale = 10)
print('P(X > 7):', round(prob, 3))

#P(1 < X < 6)
prob = uniform.cdf(x = 6, loc = 0,  scale = 10) - uniform.cdf(x = 1, loc = 0,
                                                              scale = 10)
print('P(1 < X < 6):', prob)

#Find P(17 < X < 19) if X is uniformly distributed
from scipy.stats import uniform
#'loc': the minimum point of the distribution
#'scale': is the range of the interval
prob = uniform(loc = 15, scale = 10).cdf(19) - uniform(loc = 15, scale = 10).cdf(17)
print('The probability that bird weighs between 17 and 19 grams is:', prob)

*/


	In this example we can see that by using numpy.random.uniform() method, we are able to get the random samples from uniform distribution and return the random samples.
import numpy as np

size = (5,3) #5 rows and 3 columns

sample = np.random.uniform(0, 1, size)
print(sample)


n the problem below, we will create a dataframe which consist of probability density functions and cumulative distribution functions for the values of x, and visualize the distributions. Furthermore, we will continue to find the mean and variance of the distribution using Python.

from scipy.stats import uniform
import pandas as pd
import numpy as np

x = np.linspace(1,10,100)
unif = uniform(loc = 0, scale = 20)

#Create dataframe which consist of x, pdfs, and cdfs of uniform random variable.
df = pd.DataFrame({'x': x, 'pdfs': unif.pdf(x), 'cdfs': unif.cdf(x)})
df.head()

fig = plt.figure(figsize = (5,3))
plt.bar(df.x, df.pdfs, color = 'green')
plt.title('Uniform distribution')
plt.xlabel('X values')
plt.ylabel('pdfs')
plt.show()

fig = plt.figure(figsize = (5,3))
plt.plot(df.x, df.cdfs, color = 'maroon')
plt.title('Cumulative distribution')
plt.xlabel('X values')
plt.ylabel('cdfs')
plt.show()

mean = uniform.mean(loc = 0, scale = 20)
var = uniform.var(loc = 0, scale = 20)
std = uniform.std(loc = 0, scale = 20)

print('\nMean: ', round(mean,2), '\nVariance: ', round(var,2),
      '\nStandard deviation: ', round(std,2))



/*
Suppose we want to know the probability of a certain component lasting beyond T=10 years where T is modeled as an exponential random variable with 1λ=5 years. Then, we have
1−FX(10)=e−2≈0.135.

from scipy.stats import expon

#Calculate probability that x is less than 10 when mean rate is 5
expon.cdf(x = 10, scale = 5)

#Then the required probability is given by
prob = 1 - expon.cdf(x = 10, scale = 5)
print('The required probability is:', round(prob,3))

from scipy.stats import expon #Import required libraries
import pandas as pd



Similar to the uniform distribution above, we will create a dataframe which consist of probability density functions and cumulative distribution functions for the values of x, and visualize the distributions. Furthermore, we will continue to find the mean and variance of the distribution using Python. If λ=4, then


#Define x_values
x = np.linspace(1,20,1000)
exponential = expon(scale = 4)

#Create dataframe which consist of x, pdfs, and cdfs of uniform random variable.
df = pd.DataFrame({'x': x, 'pdfs': exponential.pdf(x), 'cdfs': exponential.cdf(x)})
df.head()

fig = plt.figure(figsize = (5,3))
plt.plot(df.x, df.pdfs, color = 'green')
plt.title('Exponential distribution')
plt.xlabel('X values')
plt.ylabel('pdfs')
plt.show()


fig = plt.figure(figsize = (5,3))
plt.plot(df.x, df.cdfs, color = 'maroon')
plt.title('Cumulative distribution')
plt.xlabel('X values')
plt.ylabel('cdfs')
plt.show()


*/




Similar to the distributions above, we will create a dataframe which consist of probability density functions and cumulative distribution functions for the values of x, and visualize the distributions. Furthermore, we will continue to find the mean and variance of the distribution using Python. If μ=3,σ=2, then


from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Define x values and find pdfs and cdfs of normal distribution
x = np.linspace(-20,20,1000)
normal = norm(loc = 3, scale = 2) #Where loc = mean, and scale = standard deviation
df = pd.DataFrame({'x': x, 'pdfs': normal.pdf(x), 'cdfs': normal.cdf(x)})
df.head()

fig = plt.figure(figsize = (5,3))
plt.plot(df.x, df.pdfs, color = 'green')
plt.title('Normal(Gaussian) distribution')
plt.xlabel('X values')
plt.ylabel('pdfs')
plt.show()

fig = plt.figure(figsize = (5,3))
plt.plot(df.x, df.cdfs, color = 'maroon')
plt.title('Cumulative distribution of Normal RV')
plt.xlabel('X values')
plt.ylabel('cdfs')
plt.show()


Furthermore, find mean, variance, and standard deviation of the distribution.

mean = norm.mean(loc = 3, scale = 2)
var = norm.var(loc = 3, scale = 2)
std = norm.std(loc = 3, scale = 2)

print('\nMean: ', round(mean,2), '\nVariance: ', round(var,2),
      '\nStandard deviation: ', round(std,2))


/*
Suppose the average height of male students in a university is 175cm with a standard deviation of 6cm. What is the probability that a randomly selected male student is between 170cm and 180cm tall?

#Using python
from scipy.stats import norm
mean  = 175
sd = 6

#Calculate z-score
z1 = (170 - mean) / sd
z2 = (180 - mean) / sd

#Calculate probability
prob = norm.cdf(z2) - norm.cdf(z1)
print('The required probability:', prob)


Suppose we have data of the heights of adults in a town and the data follows a normal distribution, we have a sufficient sample size with mean equals 5.3 and the standard deviation is 1.

    Probability of height to be under 4.5 ft.
    Probability that the height of the person will be between 6.5 and 4.5 ft.


#Probability of height to be under 4.5 ft.
prob_1 = norm(loc = 5.3 , scale = 1).cdf(4.5)
print('Prob. of height under 4.5ft:', prob_1)

#probability that the height of the person will be between 6.5 and 4.5 ft.

cdf_upper_limit = norm(loc = 5.3 , scale = 1).cdf(6.5)
cdf_lower_limit = norm(loc = 5.3 , scale = 1).cdf(4.5)

prob_2 = cdf_upper_limit - cdf_lower_limit
print('Prob. of in between 6.5ft and 4.5ft:', prob_2)
*/




Example 4: First, load the built-in dataset 'tips' from the seaborn library and standardize the 'tip' column using standard normal distribution. Next, assume a random customer visits the restaurant, what is the probability that the customer will tip between $4 and $8?
#Using python
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt

#Load dataset
data = sns.load_dataset('tips')
data.head()

df = data['tip']

#Standardize the 'tip' column
normalized = (df - df.mean()) / (df.std())
normalized.head()

z1 = (4 - df.mean()) / (df.std())
z2 = (8 - df.mean()) / (df.std())

#Required probability P(4 < X < 8) or P(z1 < Z < z2)
prob = norm.cdf(z2) - norm.cdf(z1)
print('The prob. that the customer will tip between $4 to $8:', prob)
