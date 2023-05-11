# Define the first class
class LargeClass1:
    def __init__(self):
        self.data = [x for x in range(1000000)]

    def method1(self):
        result = 0
        for i in range(len(self.data)):
            result += self.data[i]
        return result

    def method2(self):
        result = 0
        for i in range(len(self.data)):
            if self.data[i] % 2 == 0:
                result += self.data[i]
        return result

# Define the second class
class LargeClass2:
    def __init__(self):
        self.data = [x**2 for x in range(1000000)]

    def method1(self):
        result = 0
        for i in range(len(self.data)):
            if self.data[i] < 500000000:
                result += self.data[i]
        return result

    def method2(self):
        result = 0
        for i in range(len(self.data)):
            if self.data[i] % 3 == 0:
                result += self.data[i]
        return result
