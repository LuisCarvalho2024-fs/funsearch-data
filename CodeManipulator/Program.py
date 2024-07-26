from CodeManipulator.Function import Function
import dataclasses

@dataclasses.dataclass(frozen=False)
class Program:
  """A parsed Python program."""
  
  before_functions: str
  functions: list[Function]

  def __str__(self) -> str:
    program = f'{self.before_functions}\n' if self.before_functions else ''
    program += '\n'.join([str(f) for f in self.functions])
    return program

  def find_function_index(self, function_name: str) -> int:
    """Returns the index of input function name."""
    function_names = [f.name for f in self.functions]
    count = function_names.count(function_name)
    if count == 0:
      raise ValueError(f'function {function_name} does not exist in program:\n{str(self)}')
    if count > 1:
      raise ValueError(f'function {function_name} exists more than once in program:\n'f'{str(self)}')
    index = function_names.index(function_name)
    return index

  def get_function(self, function_name):
    index = self.find_function_index(function_name)
    return self.functions[index]