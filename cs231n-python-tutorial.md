# CS231n Python Tutorial With Google Colab

> [CS231n](https://cs231n.github.io/)的python入门教程。

[Python Numpy Tutorial (with Jupyter and Colab) (cs231n.github.io)](https://cs231n.github.io/python-numpy-tutorial/)

[colab-tutorial.ipynb - Colaboratory (google.com)](https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/python-colab.ipynb)

## Basics of Python

```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3, 6, 8, 10, 1, 2, 1]))
```

### Basic data types

#### Numbers

```python
x = 3	# 整数
y = 2.5	# 浮点数
```

#### Booleans

```python
t, f = True, False
```

#### Strings

```python
hello = 'hello' # 可以使用单引号
world = "hello" # 双引号也行

hw12 = '{} {} {}'.format(hello, world, 12)
print(hw12) # hello world 12
```

```python
s = "hello"
print(s.capitalize())	# Capitalize a string
print(s.upper())		# Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))		# Right-justify a string, padding with spaces
print(s.center(7))		# Center a string, padding with spaces
print(s.replace('l', '(ell)')) # Replace all instances of one substring with another
print('  world '.strip()) # Strip leading and trailing whitespace
```

```
Hello
HELLO
  hello
 hello 
he(ell)(ell)o
world
```

### Containers

数据结构。

#### Lists

```python
xs = [3, 1, 2]	# Create a list
print(xs, xs[2])
print(xs[-1])	# Negative indices count from the end of the list; prints "2"
```

```python
xs[2] = 'foo'		# Lists可以存储不同的数据类型

xs.append('bar')	# 向Lists的末尾添加元素

xs.pop()			# 移除Lists的最后一个元素
```

#### Slicing

```python
nums = list(range(5)) # range is a built-in function that creates a list of integers
print(nums)			# Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])	# Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])     # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(nums[:2])     # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:])      # Get a slice of the whole list; prints ["0, 1, 2, 3, 4]"
print(nums[:-1])    # Slice indices can be negative; prints ["0, 1, 2, 3]"
nums[2:4] = [8, 9]	# Assign a new sublist to a slice
print(nums)         # Prints "[0, 1, 8, 9, 4]"
```

#### Loops

```python
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
```

可以使用`enumerate`得到元素的下标：

```python
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal))
```

#### List comprehensions:

```python
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)
# [0, 1, 4, 9, 16]
```

也可以在里面添加条件语句：

```python
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)
# [0, 4, 16]
```

#### Dictionaries

```python
d = {'cat': 'cute', 'dog': 'furry'} # Create a new dictionary with some data
print(d['cat'])		# Get an entry from a dictionary; prints "cute"
print('cat' in d)	# Check if a dictionary has a given key; prints "True"
```

在字典中添加键值对：

```python
d['fish'] = 'wet'	# Set an entry in a dictionary
print(d['fish'])	# Prints "wet"
```

如果没有这个键，会报错：

```python
print(d['monkey'])
# KeyError: 'monkey' not a key of d
```

可以使用`get()`方法来得到键对应的值：

```python
print(d.get('monkey', 'N/A'))	# Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))		# Get an element with a default; prints "wet"
```

可使用`del`关键字删除键值对：

```python
del d['fish'] # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"
```

可通过字典的`items()`方法来同时获得键和值：

```python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A {} has {} legs'.format(animal, legs))
```

可以使用`{}`构建一个字典：

```python
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)
# {0: 0, 2: 4, 4: 16}
```

#### Sets

```python
animals = {'cat', 'dog'}
print('cat' in animals)		# Check if an element is in a set; prints "True"
print('fish' in animals)	# prints "False"
```

可使用`add()`方法向集合中添加元素：

```python
animals.add('fish')	# Add an element to a set
print('fish' in animals)
print(len(animals))	# Number of elements in a set;
# 3
```

可使用`remove()`方法删除集合中的元素：

```python
animals.add('cat')		# Adding an element that is already in the set does nothing
print(len(animals))
# 3
animals.remove('cat')	# Remove an element from a set
print(len(animals))
# 2
```

在遍历集合中的元素时，使用的语法结构和Lists是一样的，但是不能确定集合中元素的顺序。

```python
animals = {'dog', 'fish', 'cat', 'fish'}
for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal))
```

输出为：

```
#1: cat
#2: fish
#3: dog
```

构建一个集合的方式和字典很相似，但是略有不同：

```python
from math import sqrt
print({int(sqrt(x)) for x in range(30)})
# {0, 1, 2, 3, 4, 5}
```

这也说明了集合中的元素是不重复的。

#### Tuples

元组是一个不可变的有序列表。元组在很多方面类似于列表，最重要的区别之一是元组可以作为字典的键，也可以作为集合中的元素，而列表不能。

```python
d = {(x, x + 1): x for x in range(5)} # Create a dictionary with tuple keys
t = (4, 5)			# Create a tuple
print(type(t))		# <class 'tuple'>
print(d[t])			# 4
print(d[(1, 2)])	# 1
```

```python
print(d)
# {(0, 1): 0, (1, 2): 1, (2, 3): 2, (3, 4): 3, (4, 5): 4}
```

元组是不可修改的：

```python
try:
    t[0] = 1
except Exception as e:
    print(e)
    # 'tuple' object does not support item assignment
```

### Functions

可使用`def`关键字声明一个函数：

```python
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))
# negative
# zero
# positive
```

上面的函数必须提供参数`x`，而下面是一个可选参数的例子：

```python
def hello(name, loud=False):
    if loud:
        print('HELLO, {}'.format(name.upper()))
    else:
        print('Hello, {}!'.format(name))

hello('Bob') # Hello, Bob!
hello('Fred', loud=True) # HELLO, FRED
```

### Classes

声明一个类的语句如下：

```python
class Greeter:

    # Constructor
    def __init__(self, name):
        self.name = name # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
          print('HELLO, {}'.format(self.name.upper()))
        else:
          print('Hello, {}!'.format(self.name))

g = Greeter('Fred')	# Construct an instance of the Greeter class
g.greet()			# Call an instance method; prints "Hello, Fred"
g.greet(loud=True)	# Call an instance method; prints "HELLO, FRED!"
```

由于我对C++比较熟悉，所以我会将python和C++进行比较，下面我会将我认为重要的部分节选进来：

#### 类的私有与公有

关于代码中`self`关键字的解释：[Python中self用法详解_python self_CLHugh的博客-CSDN博客](https://blog.csdn.net/CLHugh/article/details/75000104)。

`__init__()`可以理解为构造函数，在其中会完成成员变量的赋值：

```python
class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
```

注意，python中没有`public`和`private`关键字，成员变量的，所以可以在变量前加`__`，将成员变为私有：

```python
class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
        self.__test = 1

s = Student('karl', 100)
print(s.name)	# 'karl'
print(s.score)	# 100
print(s.__test)	# 'Student' object has no attribute '__test'
```

此时只能通过成员函数来访问私有成员变量了：

```python
class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
        self.__test = 1

    def get_test(self):
        return self.__test
```

```python
print(s.get_test())	# 1
```

同理，在函数名前面加上`__`，可将函数变为私有函数：

```python
class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
        self.__test = 1

    def get_test(self):
        return self.__test

    def __test_fun(self):
        print('this is __test_fun!')

    def run_test_fun(self):
        self.__test_fun()
        print('this is run_test_fun!')
```

但是需要注意的是，如果一个变量或函数的前后都有`__`，那么这是一个特殊变量，是可以直接访问的。

此时，执行：

```python
s.__test_fun()
# 'Student' object has no attribute '__test_fun'
```

但是执行：

```python
s.run_test_fun()
```

输出为：

```
this is _test_fun!
this is run_test_fun!
```

#### 类的继承

[Python入门 class类的继承 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/30239694)

先不看。

## Numpy

Numpy是Python中科学计算的核心库。它提供了一个高性能的多维数组对象，以及用于处理这些数组的工具。

### Arrays

numpy数组是所有相同类型的值网格，并由非负整数**元组**索引。维数是数组的秩；数组的形状是一个整数元组，给出数组在每个维度上的大小。

我们可以从嵌套的python列表中初始化numpy数组，并使用方括号访问元素：

```python
a = np.array([1, 2, 3])	# Create a rank 1 array
print(type(a), a.shape, a[0], a[1], a[2])
# <class 'numpy.ndarray'> (3,) 1 2 3
a[0] = 5				# Change an element of the array
print(a)
# [5 2 3]
```

创建一个多维数组：

```python
b = np.array([[1, 2, 3],[4, 5, 6]]) # Create a rank 2 array
print(b)
```

结果为：

```
[[1 2 3]
 [4 5 6]]
```

```python
print(b.shape)
# (2, 3)
print(b[0, 0], b[0, 1], b[1, 0])
# 1 2 4
```

可以使用`np.zeros()`方法来创建全0数组：

```python
a = np.zeros((2,2)) # Create an array of all zeros
print(a)
```

输出为：

```
[[0. 0.]
 [0. 0.]]
```

也可以使用`np.ones()`方法来创建全1数组：

```python
b = np.ones((1,2)) # Create an array of all ones
print(b)
# [[1. 1.]]
```

也可以指定大小和值，使用`np.full()`方法创建一个常数数组：

```python
c = np.full((2,2), 7) # Create a constant array
print(c)
```

```
[[7 7]
 [7 7]]
```

也可以使用`np.eye()`方法创建一个单位矩阵：

```python
d = np.eye(2) # Create a 2x2 identity matrix
print(d)
```

也可以使用`np.random.random()`方法创建一个指定大小的随机数数组：

```python
e = np.random.random((2,2)) # Create an array filled with random values
print(e)
```

输出为：

```python
[[0.21079603 0.06338665]
 [0.40904501 0.76629723]]
```

### Array indexing

可以使用切片的方法从大数组中得到一个小数组：

```python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]
print(b)
```

这里要注意的是，获得的小数组并没有自己的存储空间，所以对小数组的修改会影响到大数组：

```python
print(a[0, 1])	# 2
b[0, 0] = 77    # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])	# 77
```

还可以将整数索引与切片索引混合使用。但是，这样做将产生比原始数组低维度的数组。

对于这个数组：

```python
# Create the following rank 2 array with shape (3, 4)
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a)
```

```
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
```

将整数索引和切片索引混合：

```python
row_r1 = a[1, :]	# Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
row_r3 = a[[1], :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)	# [5 6 7 8] (4,)
print(row_r2, row_r2.shape) # [[5 6 7 8]] (1, 4)
print(row_r3, row_r3.shape)	# [[5 6 7 8]] (1, 4)
```

列也是相同的：

```python
# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)	# [ 2  6 10] (3,)
print()
print(col_r2, col_r2.shape)
# [[ 2]
# [ 6]
# [10]] (3, 1)
```

#### Integer array indexing

纯整数索引也有两种方式：

```python
a = np.array([[1, 2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[[0, 1, 2], [0, 1, 0]])	# [1 4 5]

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))	# [1 4 5]
```

> `a[[0, 1, 2], [0, 1, 0]]`意思是：
> `[0, 1, 2]`的第1个数0和`[0, 1, 0]`的第1个数0结合得到`(0, 0)`，
> `[0, 1, 2]`的第2个数1和`[0, 1, 0]`的第2个数1结合得到`(1, 1)`，
> `[0, 1, 2]`的第3个数2和`[0, 1, 0]`的第3个数0结合得到`(2, 0)`，
> 所以和`[a[0, 0], a[1, 1], a[2, 0]]`是一样的。
>
> 但要注意的是，`a[[0, 1, 2], [0, 1, 0]]`没有创建新的数组，而`np.array([a[0, 0], a[1, 1], a[2, 0]])`创建了新的匿名数组。
>
> ？什么阴间写法

下面是另一个从大数组中取出元素构造小数组的例子：

```python
# Create a new array from which we will select elements
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(a)
```

`a`为：

```
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
```

然后从大数组`a`中取出元素：

```python
# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"
```

> `[0, 2, 0, 1]`，分别对应了：
> `[ 1  2  3]`的1，
> `[ 4  5  6]`的6，
> `[ 7  8  9]`的7，
> `[10 11 12]`的11。

`a[np.arange(4), b]`是`b`中四个元素的地址，通过直接操作`a[np.arange(4), b]`，可以操作这四个元素：

```python
# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10
print(a)
```

输出为：

```python
[[11  2  3]
 [ 4  5 16]
 [17  8  9]
 [10 21 12]]
```

#### Boolean array indexing

```python
import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]])

bool_idx = (a > 2)	# Find the elements of a that are bigger than 2;
                    # this returns a numpy array of Booleans of the same
                    # shape as a, where each slot of bool_idx tells
                    # whether that element of a is > 2.

print(bool_idx)
```

```
[[False False]
 [ True  True]
 [ True  True]]
```

```python
# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])	# [3 4 5 6]

# We can do all of the above in a single concise statement:
print(a[a > 2])		# [3 4 5 6]
```

### Datatypes

每个numpy数组都是相同类型元素的网格。numpy提供了大量可用于构造数组的数字数据类型。numpy在创建数组时尝试猜测数据类型，但构造数组的函数通常还包括一个可选参数来显式指定数据类型。下面是一个示例：

```python
x = np.array([1, 2])		# Let numpy choose the datatype
y = np.array([1.0, 2.0])	# Let numpy choose the datatype
z = np.array([1, 2], dtype=np.int64) # Force a particular datatype

print(x.dtype, y.dtype, z.dtype)
# int32 float64 int64
```

### Array math

基本的数学函数在数组上按元素运行，既可以作为运算符重载，也可以作为numpy模块中的函数使用：

```python
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

# Elementwise sum; both produce the array
print(x + y)
print(np.add(x, y))
```

两个运算的结果都为：

```
[[ 6.  8.]
 [10. 12.]]
```

减、乘、除法、平方根同理：

```python
# Elementwise difference; both produce the array
print(x - y)
print(np.subtract(x, y))

# Elementwise product; both produce the array
print(x * y)
print(np.multiply(x, y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))
```

值得注意的是，`*`是逐元素相乘，而不是点积，其中，向量点积为：

```python
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

v = np.array([9, 10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))		# 219
print(np.dot(v, w))	# 219
```

`@`也代表了向量点积~~😅~~：

```python
print(v @ w) # 219
```

矩阵向量点积：

```python
# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))		# [29 67]
print(np.dot(x, v))	# [29 67]
print(x @ v)		# [29 67]
```

矩阵矩阵点积：

```python
# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))
print(x @ y)
```

矩阵求和：

```python
x = np.array([[1, 2],[3, 4]])

print(np.sum(x)) # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0)) # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1)) # Compute sum of each row; prints "[3 7]"
```

矩阵转置：

```python
print(x)
print("transpose\n", x.T)
```

输出为：

```
[[1 2]
 [3 4]]
transpose
 [[1 3]
 [2 4]]
```

向量转置：

```python
v = np.array([[1, 2, 3]])
print(v)
print("transpose\n", v.T)
```

输出为：

```python
[[1 2 3]]
transpose
 [[1]
 [2]
 [3]]
```

### Broadcasting

广播是一种强大的机制，允许numpy在执行算术运算时处理不同形状的数组。通常，我们有一个较小的数组和一个较大的数组，我们希望多次使用较小的数组来对较大的数组执行一些操作。

例如，假设我们要向矩阵的每一行添加一个常量向量。我们可以这样做：

```python
# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x) # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

print(y)
```

结果为：

```
[[ 2  2  4]
 [ 5  5  7]
 [ 8  8 10]
 [11 11 13]]
```

这是有效的；但是，当矩阵`x`非常大时，在python中计算显式循环可能会很慢。请注意，将向量`v`添加到矩阵`x`的每一行等效于通过垂直堆叠`v`的多个副本来形成矩阵`vv`，然后对`x`和`vv`进行元素求和。我们可以像这样实现这种方法：

```python
vv = np.tile(v, (4, 1))	# Stack 4 copies of v on top of each other
print(vv)				# Prints "[[1 0 1]
						#          [1 0 1]
						#          [1 0 1]
						#          [1 0 1]]"
```

```python
y = x + vv # Add x and vv elementwise
print(y)
```

结果为：

```
[[ 2  2  4]
 [ 5  5  7]
 [ 8  8 10]
 [11 11 13]]
```

numpy的广播机制让我们在进行这种计算时可以不用创建一个实际的大数组，而是直接用一个小向量即可，使用这种机制，下面的代码可以做和上面代码相同的事情：

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v # Add v to each row of x using broadcasting
print(y)
```

结果为：

```
[[ 2  2  4]
 [ 5  5  7]
 [ 8  8 10]
 [11 11 13]]
```

在上面的例子中，`x`的大小为`(4, 3)`，`v`的大小为`(3,)`，经过广播后，`v`会具有形状`(4, 3)`，并且这个求和是逐元素相乘。

广播有如下规则：

1. 如果数组没有相同的秩，则在较低秩数组的形状前面加上1，直到两个形状具有相同的shape。
2. 如果两个数组具有相同的shape，或者其中一个数组在该维度中的大小为1，则称它们在一个维度中兼容。
3. 如果阵列在所有维度上兼容，则可以进行广播。
4. 广播后，每个数组的shape等于两个输入数组的最大shape。
5. 在一个数组某个维度的大小为1而另一个数组某个维度的大小大于1的情况下，第一个数组会沿着这个维度进行复制。

这里是一些广播的例子：

```python
# Compute outer product of vectors
v = np.array([1, 2, 3])	# v has shape (3,)
w = np.array([4, 5])	# w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:

print(np.reshape(v, (3, 1)) * w)
```

结果为：

```
[[ 4  5]
 [ 8 10]
 [12 15]]
```

```python
# Add a vector to each row of a matrix
x = np.array([[1, 2, 3], [4, 5, 6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:

print(x + v)
```

结果为：

```
[[2 4 6]
 [5 7 9]]
```

```python
# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:

print((x.T + w).T)
```

结果为：

```
[[ 5  6  7]
 [ 9 10 11]]
```

```python
# Another solution is to reshape w to be a row vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(x + np.reshape(w, (2, 1)))
```

结果为：

```
[[ 5  6  7]
 [ 9 10 11]]
```

```python
# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
print(x * 2)
```

结果为：

```
[[ 2  4  6]
 [ 8 10 12]]
```

## Matplotlib

如果在notebook里面运行的话，需要加上`%matplotlib inline`。

### Plotting

```python
# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
```

此时的`x`是一个`<class 'numpy.ndarray'>`，并且能够画出一个正弦图像。

并且也可以在一幅图片上画出多条曲线：

```python
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
```

这样就是`sin`和`cos`一起的图像了。

### Subplots

```python
# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()
```

这是将一幅图像分为两幅，分别展示不同的内容。
