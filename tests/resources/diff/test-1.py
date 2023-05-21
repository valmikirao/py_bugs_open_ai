# Define a variable
x = 42

# A print statement saying Hello World!
print("Hello, world!")

# A while loop
while x > 0:
  print(x)
  x -= 1

# A for loop
for i in range(10):
  print(i)

# A list comprehension
# and a diff!
squares = [x**2 for x in range(10)]

# A class with a constructor and a method
class Person:
  def __init__(self, name):
      self.name = name

  def say_hello(self):
      print(f"Hello, my name is {self.name}")

# Creating an instance of the class
p = Person("John")
p.say_hello()
