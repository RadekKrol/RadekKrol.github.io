# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:23:19 2020

@author: radkr
"""

############################### Introduction to Python ####################################################

import numpy as np

###########################################################
### Python Basics
###########################################################


### Hello Python

# Print the sum of 7 and 10
print(7+10)

# Addition, subtraction
print(5 + 5)
print(5 - 5)

# Multiplication, division, modulo, and exponentiation
print(3 * 5)
print(10 / 2)
print(18 % 7)
print(4 ** 2)

# How much is your $100 worth after 7 years?
print(100*1.1**7)



### Variables and types

# calculate BMI
height = 180
weight = 72

BMI = weight/height**2
print(BMI)

# float type
type(BMI) # a number that has both an integer and fractional part,

# integer type
day = 5
type(day) # int - integer 

# string type
x = "radek"
type(x)  # str

# boolean type
z = True
type(z) # bool

# + operator works differently for different data types
2+3 # = 5
'ab' + 'cd' # = 'abcd'

# Create a variable savings
savings = 100
savings
print(savings)

# Create a variable growth_multiplier
growth_multiplier =1.1

# Calculate result
result = savings * growth_multiplier ** 7

# Print out result
print(result)

# vars
savings = 100
growth_multiplier = 1.1
desc = "compound interest"

# Assign product of growth_multiplier and savings to year1
year1 = savings * growth_multiplier

# Print the type of year1
print(type(year1))

# Assign sum of desc and desc to doubledesc
doubledesc = desc + desc
doubledesc

# Print out doubledesc
print(doubledesc)


### Type conversion

# conversion to string
str(var)

# conversion to int
int(x)

# conversion to float
float(x)

# conversion to boolean
bool(x)

# Definition of savings and result
savings = 100
result = 100 * 1.10 ** 7

# Fix the printout
print("I started with $" + str(savings) + " and now have $" + str(result) + ". Awesome!")

# Definition of pi_string
pi_string = "3.1415926"

# Convert pi_string into float: pi_float
pi_float = float(pi_string)


###########################################################
### Python Lists
###########################################################

### Python lists - list type

# list  is a compound data type, we can store various data types
# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Create list areas
areas = [hall, kit, liv, bed, bath]

# Print areas
print(areas)

# Adapt list areas
areas = ["hallway", hall, "kitchen", kit, "living room", 
liv,"bedroom", bed, "bathroom", bath]

# Print areas
print(areas)

# list type
print(type(areas))

# List of lists
house = [["hallway", hall],
         ["kitchen", kit],
         ["living room", liv],
         ["bedroom", bed],
         ["bathroom", bath]]
print(house)
print(type(house))


### Subsetting Lists

# list slicing - [start:end] --> inclusive:exclusive
# list element starts from 0!
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]
print(areas[1]) # Print out second element from areas
print(areas[-1]) # Print out last element from areas
print(areas[5]) # Print out the area of the living room

# Sum of kitchen and bedroom area: eat_sleep_area
eat_sleep_area = areas[3] + areas[-3]

# Print the variable eat_sleep_area
print(eat_sleep_area)

# Use slicing to create downstairs - first 7 elements
downstairs = areas[:6] # blank space means we slicing from start

# Use slicing to create upstairs -last 4 elements
upstairs = areas[-4:] # blank space means we slicing to the end

# Print out downstairs and upstairs
print(downstairs)
print(upstairs)


# Subsetting lists of lists
house[-1][1] # second element from last list of house list


### manipulating list


# Correct the bathroom area to  10.50
areas[-1] = 10.50

# Change "living room" to "chill zone"
areas[4] = "chill zone"
areas


### Extend list


# Add poolhouse data to areas, new list is areas_1
areas_1 = areas + ["poolhouse", 24.5]

# Add garage data to areas_1, new list is areas_2
areas_2 = areas_1 + ["garage", 15.45]



### Delete list elements


# removing elements with del(x[...])
del(areas[-4:-2]) # deletes poolhouse elements from list
areas


### Inner workings of lists

# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Create areas_copy
areas_copy = areas

# Change areas_copy,  the change also takes effect in the areas list
# That's because areas and areas_copy point to the same list
areas_copy[0] = 5.0

# Print areas
print(areas)

# If you want to prevent changes in areas_copy from also taking effect in areas
# you'll have to do a more explicit copy of the areas list with list() or by using [:]

# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Create areas_copy - 2 posibilities
areas_copy = list(areas)
areas_copy = areas[:]

# Change areas_copy
areas_copy[0] = 5.0

# Print areas
print(areas)





###########################################################
### Functions
###########################################################



### Example of functions

# max()
max()

# round()
# round(number, ndigits=None)
            # number -number we want to round
            # ndigits - precitions of rounding
help(round)    
round(1.78, None)
round(1.78, 2)

# Create variables var1 
var1 = [1, 2, 3, 4]

# Print out length of var1 - len() also works on string where counts characters
print(len(var1))



###  sorted() 


# sorted(iterable, key=None, reverse=False)
help(sorted)

# Create lists first and second
first = [11.25, 18.0, 20.0]
second = [10.75, 9.50]

# Paste together first and second: full
full = first + second

# Sort full in descending order: full_sorted
full_sorted = sorted(full, reverse=True)

# Print out full_sorted
print(full_sorted)



### Methods - functions that belongs to python objects


# method index()
areas.index("kitchen") # call method index() on areas

areas.count("kitchen") # how many "kitchen" appeared on areas list

# string methods -capitalize() & replace()
sister = "liz"
sister.capitalize() # we start string with capital letter
sister.replace("z", "sa") # we replace strings
sister.index("z")

# some methods can change the object
areas.append("sink") # we extend the list
areas

# upper() method
place = "poolhouse"
place_up = place.upper()
print(place_up)

# Print out the number of o's in place
print(place.count('o'))


# list methods
areas = [11.25, 18.0, 20.0, 10.75, 9.50]
print(areas.index(20.0)) # Print out the index of the element 20.0
print(areas.count(9.50)) # Print out how often 9.50 appears in areas

areas.append(24.5) # it change the object
areas.append(15.45) # it change the object
areas.reverse() # reverse order - it changes the object
areas



### Packages


# numpy - work with arrays
# matplotlip - data visualization
# scikit-learn - machine learning

# install packages - preferred version
import numpy as np

# we use numpy array() function
np.array([1,2,3])

# importing onnly array function from numpy package
from numpy import array
array([1,2,3]) # now we don't have to specify np before function


### math package

# Definition of radius
r = 0.43

# Import the math package
import math

# Calculate C
C = 2 * r * math.pi

# Calculate A
A = math.pi * r ** 2

# Build printout
print("Circumference: " + str(C))
print("Area: " + str(A))



### Selective import


# Definition of radius
r = 192500

# Import radians function of math package
from math import radians

# Travel distance of Moon over 12 degrees. Store in dist.
# You can calculate this as r * phi, where r is the radius and phi is the angle in radians
dist = r * radians(12)

# Print out dist
print(dist)



###########################################################
### Numpy
###########################################################


### Installing numpy

# pip3 install numpy
import numpy as np


# creating an array
height = [1.62,1.75,1.96,1.37]
weight = [34,45,234,54]
np_height = np.array(height)
np_weight = np.array(weight)
np_height
np_weight

# calculate BMI
bmi = np_weight/np_height **2
bmi


# array contains only the same type values
np.array([4,True, "string"]) # returns only strins (converting)


# python lists an numpy array do not work the same
python_list = [1,2,3]
numpy_array = np.array([1,2,3])
python_list + python_list   
numpy_array + numpy_array



### Your First NumPy Array



# Create list baseball
baseball = [180, 215, 210, 210, 188, 176, 209, 200]
# Create a numpy array from baseball: np_baseball
np_baseball = np.array(baseball)
# Print out type of np_baseball
print(type(np_baseball))
# multiple array by 4
np_baseball * 4
# creating logical values to tell which baseball > 200
test = np_baseball > 200
# print those values
np_baseball[test]


### Auto-converting



# Boolean values became integers
np.array([True, 1, 2]) + np.array([3, 4, False]) # True is converted to 1, False is converted to 0.




### Subsetting NumPy Arrays


np_baseball = np.array(baseball) # create array
# Print out the weight at index 50
print(np_baseball[4])
# Print out sub-array of np_basebal: index 2 up to and including index 5
print(np_baseball[2:6]) # takie indeksy są prawostronnie otwarte, jeli chcemy uwzględnić 5 indeks, musimy napisać 6




### 2D Numpy Arrays




# type of numpy arrays 
np_height = np.array([1.62,1.75,1.96,1.37])
np_weight = np.array([34,45,234,54])
type(np_height) # numpy.ndarray - numpy. tells that it is defined from numpy package (nd stands for n dimensional)
type(np_weight) # numpy.ndarray - numpy. tells that it is defined from numpy package


# shape attribute
np_2d =np.array([[4,63,65,21], [234,542,765,212]])
np_2d
np_2d.shape # 2 rows, 4 columns 

# Array contains only one type of data, when we change one element to string, the rest will be strings
np_2ds =np.array([[4,63,65,21], [234,542,765,"212"]])
print(np_2ds)

# subsetting
np_2d[0] # first list
np_2d[1] # second list
np_2d[0][2] # third element from first list

# alternaative subsetting
np_2d[0,2] # the same result as above, value before comma specify row, after the comma specify column
np_2d[:,1:3] # 2nd and 3rd element from every row
np_2d[1,:] # every element from 2nd row

# 2D Arithmetic
np_mat = np.array([[1, 2],
                   [3, 4],
                   [5, 6]])
np_mat * 2
np_mat + np.array([10, 10])
np_mat + np_mat
conversion = np.array([0.5, 1])
print(np_mat * conversion)



### Numpy: Basic Statistics



# our data
data = np.array([[62,45], [623,435],[523,43523],[421,433], [234,2127]])
print(data)

# calculate mean
np.mean(data[:,0]) # mean for first column

# median
np.median(data[:,0])

# correlation
np.corrcoef(data[:,0],data[:,1])

# std
np.std(data[:,0])

# generating data
np.random.normal(x,y,z) # data from normal distribution
                        # x - distribution mena
                        # y - distribution sd
                        # z - number of samples
height = np.round(np.random.normal(1.75, 0.20, 5000), 2)
weight = np.round(np.random.normal(60.32, 15, 5000), 2)

# binding columns
np_city = np.column_stack((height, weight))
print(np_city)


# calculating mean and median
np_height_in = np.array(data[:,0]) # numpy array np_height_in that is equal to first column of data
print(np.mean(np_height_in)) # Print out the mean of np_height_in
print(np.median(np_height_in)) # Print out the median of np_height_in

# print results
avg = np.mean(np_height_in)
print("Average: " + str(avg)) # we have to convert number to string

# Blend it all together
heights = [184, 185, 180, 181, 187, 170, 179, 183, 186, 185, 170, 187, 183, 173]
positions = ['GK', 'M', 'A', 'D', 'M', 'D', 'M', 'M', 'M', 'A', 'M', 'M', 'A', 'A']  
np_positions = np.array(positions) # convert to numpy array
np_heights = np.array(heights)
gk_heights = np.median(np_heights[np_positions == 'GK']) # Heights of the goalkeepers: gk_heights
other_heights = np.median(np_heights[np_positions != 'GK']) # Heights of the other players: other_heights
print("Median height of goalkeepers: " + str(gk_heights)) # Print out the median height of goalkeepers.
print("Median height of other players: " + str(other_heights))   # Print out the median height of other players


