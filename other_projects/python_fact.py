## How to structure classes, use __init__, and implement basic functionality

class MyClass:
    def __init__(self, value):
        self.value = value

    def display_value(self):
        print(f"The value is: {self.value}")
# Example usage
my_instance = MyClass(10)
my_instance.display_value() # Output: The value is: 10




class myclass_read():
    def __init__(self,value):
        self.value = va

# # Example of a class with a method that returns a formatted string
# class Person:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age

#     def get_info(self):
#         return f"{self.name} is {self.age} years old."
# # Example usage
# person_instance = Person("Alice", 30)
# print(person_instance.get_info())  # Output: Alice is 30 years old.
# # Example of a class with a method that modifies an attribute
# class Counter:
#     def __init__(self):
#         self.count = 0

#     def increment(self):
#         self.count += 1

#     def get_count(self):
#         return self.count   
# # Example usage
# counter_instance = Counter()
# counter_instance.increment()
# print(counter_instance.get_count())  # Output: 1


#Task: Write a class Portfolio that stores multiple Stock objects and calculates total value.
class Stock:
    def __init__(self, symbol, shares, price_per_share):
        self.symbol = symbol
        self.shares = shares
        self.price_per_share = price_per_share

    def get_total_value(self):
        return self.shares * self.price_per_share

class Portfolio:
    def __init__(self):
        self.stocks = []

    def add_stock(self, stock):
        self.stocks.append(stock)

    def get_total_value(self):
        return sum(stock.get_total_value() for stock in self.stocks)
    
# Example usage
portfolio = Portfolio() 
stock1 = Stock("AAPL", 10, 150.00)
stock2 = Stock("GOOGL", 5, 2800.00)
portfolio.add_stock(stock1)
portfolio.add_stock(stock2)
print(f"Total portfolio value: ${portfolio.get_total_value():.2f}")  # Output: Total portfolio value: $14500.00


class grosary:
    def __init__(self,item_name,item_prize,discount):
        self.item_name = item_name
        self.item_prize = item_prize
        self.discount = discount
        
    def total_discount(self):
        return (self.item_prize/self.discount)

class basket:
    def __init__(self):
        self.basket= []
    def add_basket(self,grosary):
        self.basket.append(grosary)

    def total_discount(self):
        return sum(grosary.total_discount() for grosary in self.basket)

basket = basket()
item1 = grosary("oil",500,10)
item2 = grosary("biscuts",150,10)
basket.add_basket(item1)
basket.add_basket(item2)
print(f"Total discount:{basket.total_discount()}")

import pandas as pd

data = {'symbol': ['AAPL', 'GOOG', 'MSFT'], 'price': [150, 2800, 300]}
df = pd.DataFrame(data)

# Add a column
df['price_with_tax'] = df['price'] * 1.18
print(df)

# Filter
expensive = df[df['price'] > 1000]
print(expensive)


df1 = {'user_names':['hameed','khan','mohammad'],'employe_id':[101,102,103],'date_of_join':[2021,2022,2023]}
dff = pd.DataFrame(df1)
print(dff)

year_filter = dff[dff['date_of_join'] > 2021]
print(year_filter)

#read data from an excel file and covert it in to json format
# df2 = pd.read_excel('data.xlsx')
# json_data = df2.to_json(orient='records')
# print(json_data)
# # Save DataFrame to CSV
# df2.to_csv('data.csv', index=False)
# # Read CSV file
# df3 = pd.read_csv('data.csv')  
# print(df3)
# # Convert DataFrame to JSON
# json_data = df3.to_json(orient='records')
# print(json_data)

# Create a DataFrame with Stock symbols Closing prices over 5 days Then calculate: Daily percentage change Average price
df_stocks = pd.DataFrame({
    'symbol': ['AAPL', 'GOOG', 'MSFT'],
    'day1': [150, 2800, 300],
    'day2': [155, 2850, 310],
    'day3': [160, 2900, 320],
    'day4': [165, 2950, 330],
    'day5': [170, 3000, 340]
})
# Calculate daily percentage change
df_stocks['daily_pct_change'] = df_stocks[['day1', 'day2', 'day3', 'day4', 'day5']].pct_change(axis=1).mean(axis=1)
# Calculate average price
df_stocks['average_price'] = df_stocks[['day1', 'day2', 'day3', 'day4', 'day5']].mean(axis=1)
print(df_stocks[['symbol', 'daily_pct_change', 'average_price']])



# program to identify the parent node of a given node in a tree structure
class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parent = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def get_parent(self):
        return self.parent
    
# Example usage
root = Node("Root")
child1 = Node("Child 1")
child2 = Node("Child 2")
root.add_child(child1)
root.add_child(child2)  
print(f"Parent of {child1.name} is {child1.get_parent().name}")  # Output: Parent of Child 1 is Root

# Program to identify child nodes of a given node in a tree structure
def get_child_nodes(node):
    return node.children

# Example usage
children_of_root = get_child_nodes(root)
print(f"Children of {root.name} are {[child.name for child in children_of_root]}")
# Output: Children of Root are ['Child 1', 'Child 2']

# programmes on oops concepts
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclasses must implement this method")

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"
# Example usage
dog = Dog("Buddy")  
cat = Cat("Whiskers")
print(f"{dog.name} says: {dog.speak()}")  # Output: Buddy says: Woof!
print(f"{cat.name} says: {cat.speak()}")  # Output: Whiskers says: Meow!


# Program to demonstrate inheritance and method overriding
class Vehicle:
    def __init__(self, brand):
        self.brand = brand

    def start(self):
        return f"{self.brand} vehicle is starting." 
class Car(Vehicle):
    def start(self):
        return f"{self.brand} car is starting with a roar!"
class Bike(Vehicle):
    def start(self):
        return f"{self.brand} bike is starting with a vroom!"
# Example usage
car = Car("Toyota")
bike = Bike("Yamaha")
Vehicle = Vehicle("Generic")
print(car.start())  # Output: Toyota car is starting with a roar!
print(bike.start())  # Output: Yamaha bike is starting with a vroom!
print(Vehicle.start())  # Output: Generic vehicle is starting.

# Program to demonstrate encapsulation
class BankAccount:
    def __init__(self, account_number, balance=0):
        self.__account_number = account_number  # Private attribute
        self.__balance = balance  # Private attribute

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"Deposited: ${amount}. New balance: ${self.__balance}")
        else:
            print("Deposit amount must be positive.")

    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            print(f"Withdrew: ${amount}. New balance: ${self.__balance}")
        else:
            print("Invalid withdrawal amount.")

    def get_balance(self):
        return self.__balance
# Example usage
account = BankAccount("123456789", 1000)    
account.deposit(500)  # Output: Deposited: $500. New balance: $1500
account.withdraw(200)  # Output: Withdrew: $200. New balance: $1300 
print(f"Current balance: ${account.get_balance()}")  # Output: Current balance: $1300


# Program to demonstrate polymorphism       
class Shape:
    def area(self):
        raise NotImplementedError("Subclasses must implement this method")
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height
# Example usage
shapes = [Circle(5), Rectangle(4, 6)]
for shape in shapes:
    print(f"Area of {shape.__class__.__name__} is: {shape.area()}")
# Output:
# Area of Circle is: 78.5
# Area of Rectangle is: 24
# Program to demonstrate method overloading

# Try this without using [::-1].
def reverse_words(s: str) -> str:
    words = s.strip().split()
    reversed_words = []
    for word in words:
        reversed_words.insert(0, word)
    return ' '.join(reversed_words)
# Example usage
print(reverse_words("Hello World"))  # Output: "World Hello"

def reverse_of_string(s):
    # Filter and normalize characters: only alphanumeric, all lowercase
    filtered = []
    for ch in s:
        if '0' <= ch <= '9' or 'a' <= ch.lower() <= 'z':
            filtered.append(ch.lower())
    # Reverse the filtered string
    return ''.join(reversed(filtered))

print(reverse_of_string('Khan'))

# Function to check if two strings are anagrams
def is_anagram(s1: str, s2: str) -> bool:
    return sorted(s1) == sorted(s2)
# Example usage
print(is_anagram("listen", "silent"))  # Output: True
print(is_anagram("hello", "world"))    # Output: False

# try using colloctions.Counter for anagram check
from collections import Counter
def is_anagram_counter(s1: str, s2: str) -> bool:
    return Counter(s1) == Counter(s2)


# Return indices of two numbers such that they add up to target.
def two_sum(numbers: list, target: int) -> tuple:
    num_dict = {}
    for i, num in enumerate(numbers):
        complement = target - num
        if complement in num_dict:
            return (num_dict[complement], i)
        num_dict[num] = i
    return None

# Example usage
print(two_sum([2, 7, 11, 15], 9))  # Output: (0, 1)

# EASIER METHOD 1: Brute Force (Simple nested loops)
def two_sum_easy(numbers: list, target: int) -> tuple:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):  # Start from i+1 to avoid same element
            if numbers[i] + numbers[j] == target:
                return (i, j)
    return None

print(two_sum_easy([2, 7, 11, 15], 9))  # Output: (0, 1)

# EASIER METHOD 2: Using list methods
def two_sum_simple(numbers: list, target: int) -> tuple:
    for i, num in enumerate(numbers):
        needed = target - num
        # Look for the needed number in the rest of the list
        for j in range(i + 1, len(numbers)):
            if numbers[j] == needed:
                return (i, j)
    return None

print(two_sum_simple([2, 7, 11, 15], 9))  # Output: (0, 1)

# Function to check if a string has valid parentheses
def is_valid(s: str) -> bool:
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in mapping:
            if not stack or stack.pop() != mapping[ch]:
                return False
        else:
            stack.append(ch)
    return not stack
# Example usage
print(is_valid("()"))          # Output: True
print(is_valid("([{}])"))      # Output: True
print(is_valid("(]"))          # Output: False
print(is_valid("([)]"))        # Output: False
print(is_valid("{[]}"))       # Output: True

# Function to find the first unique character in a string
#without using collections.Counter
def first_unique(s: str) -> int:
    char_count = {}
    for ch in s:
        char_count[ch] = char_count.get(ch, 0) + 1
    for i, ch in enumerate(s):
        if char_count[ch] == 1:
            return i
    return -1
# Example usage
print(first_unique("leetcode"))  # Output: 0
print(first_unique("loveleetcode"))  # Output: 2

# EASIER APPROACH 1: Using string.count() method
def first_unique_easy(s: str) -> int:
    for i, ch in enumerate(s):
        if s.count(ch) == 1:  # Count how many times this character appears
            return i
    return -1

print(first_unique_easy("leetcode"))     # Output: 0
print(first_unique_easy("loveleetcode")) # Output: 2

# EASIER APPROACH 2: Using collections.Counter (simpler)
from collections import Counter
def first_unique_counter(s: str) -> int:
    count = Counter(s)  # Automatically counts all characters
    for i, ch in enumerate(s):
        if count[ch] == 1:
            return i
    return -1

print(first_unique_counter("leetcode"))     # Output: 0
print(first_unique_counter("loveleetcode")) # Output: 2

# EASIER APPROACH 3: One-liner using next() function
def first_unique_oneliner(s: str) -> int:
    return next((i for i, ch in enumerate(s) if s.count(ch) == 1), -1)

print(first_unique_oneliner("leetcode"))     # Output: 0
print(first_unique_oneliner("loveleetcode")) # Output: 2

# Function to find the longest palindromic substring
def longest_palindrome(s: str) -> str:
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l+1:r]

    result = ''
    for i in range(len(s)):
        odd = expand(i, i)
        even = expand(i, i+1)
        result = max(result, odd, even, key=len)
    return result
# Example usage
print(longest_palindrome("babad"))  # Output: "bab" or "aba"
# EASIER METHOD: Using dynamic programming
def longest_palindrome_dp(s: str) -> str:
    n = len(s)
    if n == 0:
        return ""
    
    dp = [[False] * n for _ in range(n)]
    start, max_length = 0, 1
    
    for i in range(n):
        dp[i][i] = True  # Single character is a palindrome
    
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_length = 2
    
    for length in range(3, n + 1):  # Length of substring
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_length = length
    
    return s[start:start + max_length]



# aws lambda function to get the longest palindromic substring
def lambda_handler(event, context):
    s = event.get("input", "")
    result = longest_palindrome(s)
    return {
        "statusCode": 200,
        "body": result
    }
# Example usage
event = {"input": "babad"}
context = {}
response = lambda_handler(event, context)
print(response)



#List & Dict Comprehensions
nums = [1, 2, 3, 4]
squares = [x*x for x in nums if x % 2 == 0]  
# Output: [4, 16]

my_dict = {x: x**2 for x in range(5)}
# Output: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

#identify the repetative letters in the string 'Hellow'
string = 'good'
repetitive_letters = {ch: string.count(ch) for ch in set(string) if string.count(ch) > 1}
print(repetitive_letters)


# Function Definitions
def greet(name, age=25):
    print(f"Hello {name}, age {age}  #########")

def add(*args):
    print("Adding:", args)
    return print(sum(args))

def print_details(**kwargs):
    for k, v in kwargs.items():
        print(k, v)

# Example usage
greet("Alice")        # Output: Hello Alice, age 25
result1 = add(1, 2, 3)          # Output: 6
print_details(name="Bob", age=30, city="New York")



# oops concepts

class Car:
    wheels = 4

    def __init__(self, brand):
        self.brand = brand

    @staticmethod
    def honk():
        print("Beep Beep!")

    @classmethod
    def vehicle_type(cls):
        return "Car"

# Example usage
car1 = Car("Tesla")
print(car1.brand)
print(car1.vehicle_type())
Car.honk()

# Decorators

def uppercase_decorator(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper()
    return wrapper

@uppercase_decorator
def greet(name):
    return f"Hello {name}"

print(greet("Alice In the wonderland"))


#reverse of a string
def reverse_string(s):
    return s[::-1]

print(reverse_string("Hello World"))

#reverse of a string with out slicing
def reverse_string_no_slicing(s):
    reversed_str = ""
    for char in s:
        reversed_str = char + reversed_str
    return reversed_str

print(reverse_string_no_slicing("Hello World"))


#Flatten a Nested List
def flatten_list(nested_list):
    flat_list = []
    for sublist in nested_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list

print(flatten_list([[1, 2], [3, 4], [5]]))

#Most Frequent Word with simplified code

def most_frequent_word(text):
    words = text.split()
    frequency = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1
    most_frequent = max(frequency, key=frequency.get)
    return most_frequent

print(most_frequent_word("hello world hello everyone hello world"))  # Output: "hel

#Rotate Array
def rotate_array(arr, k):
    n = len(arr)
    k = k % n  # Handle cases where k > n
    return arr[-k:] + arr[:-k]

print(rotate_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 2))  # Output: [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def rotate_array_loop(arr, k):
    n = len(arr)
    k = k % n
    result = []
    
    # Add last k elements first
    for i in range(n - k, n):
        result.append(arr[i])
    
    # Add first n-k elements
    for i in range(n - k):
        result.append(arr[i])
    
    return result

print(rotate_array_loop([1, 2, 3, 4, 5], 2))  # Output: [4, 5, 1, 2, 3]


# CSV File Handling Practice
import csv

def write_to_csv(filename, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def read_from_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        return list(reader)

def read_from_file(filename):
    with open(filename, 'r') as file:
        return file.read()

# Example usage
write_to_csv('data.csv', [['Name', 'Age'], ['Alice', 30], ['Bob', 25]])
print(read_from_csv('data.csv'))


#Reverse words in a sentence: "I love Python" → "Python love I"

def reverse_words(sentence):
    words = sentence.split()
    words.reverse()
    return ' '.join(words)

print(reverse_words("I love Python"))  # Output: "Python love I"

#Find the second largest number in a list.
def second_largest(numbers):
    first = second = float('-inf')
    for n in numbers:
        if n > first:
            second = first
            first = n
        elif first > n > second:
            second = n
    return second if second != float('-inf') else None

def first_largest(numbers):
    first = float('-inf')
    for n in numbers:
        if n > first:
            first = n
    return first if first != float('-inf') else None

print(second_largest([1, 2, 3, 4, 5]))  # Output: 4
print(first_largest([1, 2, 3, 4, 5]))   # Output: 5


#Count vowels in a string.
def count_vowels(s):
    vowels = "aeiouAEIOU"
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count

print(f"count_vowels",count_vowels('Hello Vowels'))  # Output: 3

#Merge two dictionaries.
def merge_dictionaries(dict1, dict2):
    merged = dict1.copy()  # Start with the first dictionary
    merged.update(dict2)    # Update with the second dictionary
    return merged

dict_a = {'a': 1, 'b': 2}
dict_b = {'b': 3, 'c': 4}
print(merge_dictionaries(dict_a, dict_b))  # Output: {'a': 1, 'b': 3, 'c': 4}

#check number is prime
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

print(is_prime(11))  # Output: True
print(is_prime(4))   # Output: False


#binary search
def binary_search(arr, target):
    l, r = 0, len(arr)-1
    while l <= r:
        mid = (l+r)//2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            l = mid+1
        else:
            r = mid-1
    return -1

print(binary_search([1,2,3,4,5], 4))  # 3

#sorting and searching
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

print(bubble_sort([64, 34, 25, 12, 22, 11, 90]))  # Sorted array

#remove duplicates in list
def remove_duplicates(lst):
    return list(set(lst))


print(remove_duplicates([1, 2, 2, 3, 4, 4, 5]))  # Output: [1, 2, 3, 4, 5]

# identify a item in list and return its index
def find_item_index(lst, item):
    try:
        return lst.index(item)
    except ValueError:
        return -1
    
print(find_item_index([1, 2, 3, 4, 5], 3))  # Output: 2
print(find_item_index([1, 2, 3, 4, 5], 6))  # Output: -1

#identify the highest value in the list and its index
def find_highest_value_and_index(lst):
    if not lst:
        return None, -1
    highest = lst[0]
    index = 0
    for i, value in enumerate(lst):
        if value > highest:
            highest = value
            index = i
    return highest, index

print(find_highest_value_and_index([1, 2, 3, 4, 5]))  # Output: (5, 4)

#identify the highest repeated value in the list and its index
def find_highest_repeated_value_and_index(lst):
    if not lst:
        return None, -1
    value_counts = {}
    for i, value in enumerate(lst):
        if value in value_counts:
            value_counts[value].append(i)
        else:
            value_counts[value] = [i]
    highest_repeated = None
    highest_index = -1
    for value, indices in value_counts.items():
        if len(indices) > 1:  # Check if the value is repeated
            if highest_repeated is None or value > highest_repeated:
                highest_repeated = value
                highest_index = indices[0]  # Return the first occurrence index
    return highest_repeated, highest_index

print(find_highest_repeated_value_and_index([1, 2, 3, 4, 5, 3, 2]))  # Output: (3, 2)