import cmake_example

a = 1
b = 2
# normal
print("{} + {} = {}".format(a, b, cmake_example.add(a, b)))

# lambda
print("{} - {} = {}".format(a, b, cmake_example.subtract(a, b)))

# keyword 
print("{} * {} = {}".format(a, b, cmake_example.multiply(i=a, j=b)))
print("{} * {} = {}".format(b, a, cmake_example.multiply(j=a, i=b)))

# default
print("{} / {} = {}".format(a, b, cmake_example.divide(i=a, j=b)))
print("{} / {} = {}".format(b, a, cmake_example.divide(j=a, i=b)))
print("{} / {} = {}".format(a, None, cmake_example.divide(a)))
