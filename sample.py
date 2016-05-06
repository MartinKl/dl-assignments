from blocks.serialization import load
from blocks.filter import VariableFilter

f = open('model.bin', 'rb')
main_loop = load(f)
model = main_loop.model
print(dir(main_loop.model()))