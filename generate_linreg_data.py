import sys
from random import random


def gen1(A: int, B: int, noise_radius: float):
  def func(x):
    noise = random() * noise_radius * 2 - noise_radius
    return A * x + B + noise
  
  file_name = 'data/%sx_plus_%s.csv' % (A, B)  
  f = open(file_name, 'w')
  
  for x in range(10000):
    if random() > .7:
      f.write('%s,%s\n' % (x, func(x)))
  
  f.close()
  print('done generating %s' % file_name)


def print_usage():
  print('Usage: python %s [<int> <int> [<float>]]' % sys.argv[0])


a = 3
b = 9
noise_radius = 2.5

if len(sys.argv) >= 3:
  try:
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    if len(sys.argv) > 3:
      noise_radius = float(sys.argv[3])
  except ValueError as ex:
    print('illegal arguments: %s' % str(ex))
    print_usage()
    quit(1)
elif len(sys.argv) != 1:
  print('incorrect number of arguments')
  print_usage()
else:
  print('using defaults of A=%s, B=%s, noise_factor=%s' %
        (a, b, noise_radius))

gen1(A=a, B=b, noise_radius=noise_radius)

