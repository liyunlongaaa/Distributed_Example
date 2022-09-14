import os

print('RANK' in os.environ and 'WORLD_SIZE' in os.environ)
print(os.environ['WORLD_SIZE'])

