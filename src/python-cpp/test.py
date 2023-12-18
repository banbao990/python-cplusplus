import python_example

a = 1
b = 2
# normal
print("{} + {} = {}".format(a, b, python_example.add(a, b)))

# lambda
print("{} - {} = {}".format(a, b, python_example.subtract(a, b)))

# keyword 
print("{} * {} = {}".format(a, b, python_example.multiply(i=a, j=b)))
print("{} * {} = {}".format(b, a, python_example.multiply(j=a, i=b)))

# default
print("{} / {} = {}".format(a, b, python_example.divide(i=a, j=b)))
print("{} / {} = {}".format(b, a, python_example.divide(j=a, i=b)))
print("{} / {} = {}".format(a, None, python_example.divide(a)))
