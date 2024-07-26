import dataclasses

# Everything is string
@dataclasses.dataclass
class Function:
  """A parsed Python function."""
  name: str
  args: str
  body: str
  return_type: str = None
  docstring: str = None

  def __str__(self):
    return_type = f' -> {self.return_type}' if self.return_type else ''

    function = ''
    if self.docstring:
      function += f'"""{self.docstring}"""\n'
    # self.body is already indented.
    function += f'def {self.name}({self.args}){return_type}:\n'
    function += self.body + '\n'
    return function

  def __setattr__(self, name: str, value: str) -> None:
    # Ensure there aren't leading & trailing new lines in `body`.
    if name == 'body':
      value = value.strip('\n')
    # Ensure there aren't leading & trailing quotes in `docstring``.
    if name == 'docstring' and value is not None:
      if '"""' in value:
        value = value.strip()
        value = value.replace('"""', '')
    super().__setattr__(name, value)