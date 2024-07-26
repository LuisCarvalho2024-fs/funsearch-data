import dataclasses

@dataclasses.dataclass(frozen=True)
class Prompt:
  code: str
  version_generated: int
  island_id: int