import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from itertools import product

from tabulate import tabulate


class Operator(Enum):
  AND = 'and'
  IMPLIES = 'implies'
  NOT = 'not'
  OR = 'or'


class Formula(ABC):
  @abstractmethod
  def __repr__(self) -> str:
    pass

  @abstractmethod
  def substitute(self, substitution: t.Dict[str, 'Formula']) -> 'Formula':
    pass

  @abstractmethod
  def free_variables(self) -> set:
    pass


@dataclass
class Proposition(Formula):
  value: t.Union[str, bool]

  def __hash__(self):
    return hash(self.value)

  def __repr__(self) -> str:
    return str(self.value)

  def substitute(self, substitution: t.Dict[str, Formula]) -> Formula:
    if isinstance(self.value, str) and self.value in substitution:
      return substitution[self.value]
    return self

  def free_variables(self) -> set:
    return {self.value} if isinstance(self.value, str) else set()


@dataclass
class Expression(Formula):
  operator: Operator
  operands: t.Tuple[t.Union['Expression', Proposition], ...]

  def __hash__(self):
    return hash((self.operator, self.operands))

  def __repr__(self):
    if self.operator == Operator.NOT:
      return f'¬{self.operands[0]}'
    elif self.operator == Operator.AND:
      return f'({self.operands[0]} ∧ {self.operands[1]})'
    elif self.operator == Operator.OR:
      return f'({self.operands[0]} ∨ {self.operands[1]})'
    elif self.operator == Operator.IMPLIES:
      return f'({self.operands[0]} → {self.operands[1]})'
    else:
      return f"Unknown({', '.join(map(str, self.operands))})"

  def substitute(self, substitution: t.Dict[str, Formula]) -> Formula:
    return Expression(
      self.operator,
      tuple(operand.substitute(substitution) for operand in self.operands),
    )

  def free_variables(self) -> set:
    return set.union(*(operand.free_variables() for operand in self.operands))


def evaluate(
  expression: t.Union[Proposition, Expression],
  substitutions: t.Dict[str, bool] = {},
) -> bool:
  """
  Evaluates a logical expression given a set of variable substitutions.

  Args:
    expression (Union[Proposition, Expression]): The logical expression to evaluate.
    substitutions (Dict[str, bool], optional): A dictionary of variable substitutions. Defaults to {}.

  Returns:
    bool: The truth value of the expression.

  Raises:
    ValueError: If an unsupported expression type is encountered.

  Examples:
    >>> p, q = Proposition('p'), Proposition('q')
    >>> expr = Expression(Operator.AND, (p, q))
    >>> evaluate(expr, {'p': True, 'q': False})
    False
    >>> evaluate(expr, {'p': True, 'q': True})
    True
  """
  match expression:
    case Proposition():
      if isinstance(expression.value, bool):
        return expression.value
      return substitutions.get(expression.value, False)
    case Expression(operator=Operator.AND, operands=operands):
      return all(evaluate(operand, substitutions) for operand in operands)
    case Expression(operator=Operator.OR, operands=operands):
      return any(evaluate(operand, substitutions) for operand in operands)
    case Expression(operator=Operator.NOT, operands=(operand,)):
      return not evaluate(operand, substitutions)
    case Expression(operator=Operator.IMPLIES, operands=(premise, conclusion)):
      return (not evaluate(premise, substitutions)) or evaluate(
        conclusion, substitutions
      )
    case _:
      raise ValueError(f'Unsupported expression type: {type(expression)}')


def get_variables(expression: t.Union[Proposition, Expression]) -> set:
  """
  Extracts all variables from a given expression.

  Args:
    expression (Union[Proposition, Expression]): The expression to extract variables from.

  Returns:
    set: A set of all variables in the expression.
  """
  match expression:
    case Proposition():
      return {expression.value} if isinstance(expression.value, str) else set()
    case _:
      return set.union(
        *(get_variables(operand) for operand in expression.operands)
      )


def is_semantic_consequence(
  premises: t.List[t.Union[Proposition, Expression]],
  conclusion: t.Union[Proposition, Expression],
) -> bool:
  """
  Determines if a conclusion is a semantic consequence of given premises.

  Args:
    premises (List[Union[Proposition, Expression]]): A list of premises.
    conclusion (Union[Proposition, Expression]): The conclusion to check.

  Returns:
    bool: True if the conclusion is a semantic consequence of the premises, False otherwise.

  Examples:
    >>> p, q = Proposition('p'), Proposition('q')
    >>> premise = Expression(Operator.IMPLIES, (p, q))
    >>> is_semantic_consequence([premise, p], q)
    True
    >>> is_semantic_consequence(
    ...   [Expression(Operator.OR, (p, q))], Expression(Operator.AND, (p, q))
    ... )
    False
  """
  variables = set.union(
    *(get_variables(p) for p in premises), get_variables(conclusion)
  )

  for values in product([True, False], repeat=len(variables)):
    substitutions = dict(zip(variables, values))

    if all(evaluate(premise, substitutions) for premise in premises):
      if not evaluate(conclusion, substitutions):
        return False

  return True


def generate_truth_table(expr: t.Union[Proposition, Expression]) -> str:
  """
  Generates a truth table for a given logical expression.

  The table rows are ordered to correspond to binary counting from 0 to 1
  for the input variables.

  Args:
    expr (Union[Proposition, Expression]): The logical expression to generate a truth table for.

  Returns:
    str: A string representation of the truth table.

  Examples:
    >>> p, q = Proposition('p'), Proposition('q')
    >>> expr = Expression(Operator.IMPLIES, (p, q))
    >>> print(generate_truth_table(expr))
    ┌───┬───┬───────────┐
    │ p │ q │ (p → q)   │
    ├───┼───┼───────────┤
    │ 0 │ 0 │ 1         │
    │ 0 │ 1 │ 1         │
    │ 1 │ 0 │ 0         │
    │ 1 │ 1 │ 1         │
    └───┴───┴───────────┘
  """
  variables = sorted(get_variables(expr))
  subexpressions = get_subexpressions(expr)

  headers = variables + [str(subexpr) for subexpr in subexpressions]

  rows = []

  for values in product([False, True], repeat=len(variables)):
    substitutions = dict(zip(variables, values))

    row = [int(substitutions[var]) for var in variables]

    for subexpr in subexpressions:
      result = evaluate(subexpr, substitutions)
      row.append(int(result))

    rows.append(row)

  return tabulate(rows, headers=headers, tablefmt='simple_outline')


def get_subexpressions(
  expr: t.Union[Proposition, Expression],
) -> t.List[t.Union[Proposition, Expression]]:
  """
  Extracts all subexpressions from a given expression.

  Args:
    expr (Union[Proposition, Expression]): The expression to extract subexpressions from.

  Returns:
    List[Union[Proposition, Expression]]: A list of all subexpressions in the given expression.
  """
  if isinstance(expr, Proposition):
    return [expr]

  subexprs = []

  for operand in expr.operands:
    subexprs.extend(get_subexpressions(operand))

  subexprs.append(expr)

  return subexprs


def _encode_graph_coloring(
  graph: t.Dict[str, t.List[str]], num_colors: int
) -> Expression:
  """
  Encodes a graph coloring problem as a logical expression.

  Args:
    graph (Dict[str, List[str]]): A dictionary representing the graph. Keys are nodes, values are lists of adjacent nodes.
    num_colors (int): The number of colors to use for coloring.

  Returns:
    Expression: A logical expression representing the graph coloring constraints.

  Examples:
    >>> graph = {'A': ['B', 'C'], 'B': ['A', 'C'], 'C': ['A', 'B']}
    >>> expr = encode_graph_coloring(graph, 3)
    >>> print(expr)
    (((A_color_0 ∨ A_color_1) ∨ A_color_2) ∧ (...))
  """
  expressions = []

  # For each node, it must have at least one color
  for node in graph:
    expressions.append(
      Expression(
        Operator.OR,
        tuple(Proposition(f'{node}_color_{i}') for i in range(num_colors)),
      )
    )

  # For each node, it can't have more than one color
  for node in graph:
    for i in range(num_colors):
      for j in range(i + 1, num_colors):
        expressions.append(
          Expression(
            Operator.OR,
            (
              Expression(Operator.NOT, (Proposition(f'{node}_color_{i}'),)),
              Expression(Operator.NOT, (Proposition(f'{node}_color_{j}'),)),
            ),
          )
        )

  # Adjacent nodes can't have the same color
  for node, neighbors in graph.items():
    for neighbor in neighbors:
      for color in range(num_colors):
        expressions.append(
          Expression(
            Operator.OR,
            (
              Expression(Operator.NOT, (Proposition(f'{node}_color_{color}'),)),
              Expression(
                Operator.NOT, (Proposition(f'{neighbor}_color_{color}'),)
              ),
            ),
          )
        )

  return Expression(Operator.AND, tuple(expressions))


def is_graph_colorable(
  graph: t.Dict[str, t.List[str]], num_colors: int
) -> bool:
  """
  Determines if a graph is colorable with a given number of colors.

  Args:
    graph (Dict[str, List[str]]): A dictionary representing the graph. Keys are nodes, values are lists of adjacent nodes.
    num_colors (int): The number of colors to use for coloring.

  Returns:
    bool: True if the graph is colorable with the given number of colors, False otherwise.

  Examples:
    >>> triangle = {'A': ['B', 'C'], 'B': ['A', 'C'], 'C': ['A', 'B']}
    >>> is_graph_colorable(triangle, 2)
    False
    >>> is_graph_colorable(triangle, 3)
    True
  """
  expression = _encode_graph_coloring(graph, num_colors)

  variables = get_variables(expression)

  for values in product([False, True], repeat=len(variables)):
    substitutions = dict(zip(variables, values))

    if evaluate(expression, substitutions):
      return True

  return False


class InferenceRule(ABC):
  @abstractmethod
  def check(self, premises: t.List[Formula]) -> bool:
    pass

  @abstractmethod
  def apply(self, premises: t.List[Formula]) -> t.Optional[Formula]:
    pass


class ModusPonens(InferenceRule):
  def check(self, premises: t.List[Formula]) -> bool:
    return (
      len(premises) == 2
      and isinstance(premises[0], Expression)
      and premises[0].operator == Operator.IMPLIES
    )

  def apply(self, premises: t.List[Formula]) -> t.Optional[Formula]:
    if self.check(premises):
      implication, antecedent = premises

      assert isinstance(implication, Expression)

      if implication.operands[0] == antecedent:
        return implication.operands[1]

    return None


class AndIntroduction(InferenceRule):
  def check(self, premises: t.List[Formula]) -> bool:
    return len(premises) == 2

  def apply(self, premises: t.List[Formula]) -> t.Optional[Formula]:
    if self.check(premises):
      return Expression(Operator.AND, premises)
    return None


class AndElimination(InferenceRule):
  def check(self, premises: t.List[Formula]) -> bool:
    return (
      len(premises) == 1
      and isinstance(premises[0], Expression)
      and premises[0].operator == Operator.AND
    )

  def apply(
    self, premises: t.List[Formula]
  ) -> t.Optional[t.Tuple[Formula, Formula]]:
    if self.check(premises):
      conjunction = premises[0]
      return conjunction.operands
    return None


class OrIntroduction(InferenceRule):
  def check(self, premises: t.List[Formula]) -> bool:
    return len(premises) == 1

  def apply(self, premises: t.List[Formula]) -> t.Optional[Formula]:
    if self.check(premises):
      return lambda other: Expression(Operator.OR, (premises[0], other))
    return None


@dataclass
class ProofStep:
  formula: Formula
  rule: t.Optional[InferenceRule]
  premises: t.List[int]


class Proof:
  def __init__(self):
    self.steps: t.List[ProofStep] = []

  def add_step(
    self,
    formula: Formula,
    rule: t.Optional[InferenceRule],
    premises: t.List[int],
  ):
    self.steps.append(ProofStep(formula, rule, premises))

  def check_validity(self) -> bool:
    for _, step in enumerate(self.steps):
      if step.rule is None:  # Premise or assumption
        continue

      premise_formulas = [self.steps[j].formula for j in step.premises]
      if (
        not step.rule.check(premise_formulas)
        or step.rule.apply(premise_formulas) != step.formula
      ):
        return False

    return True

  def display(self):
    for i, step in enumerate(self.steps):
      premises = ', '.join(map(str, step.premises))
      rule_name = step.rule.__class__.__name__ if step.rule else 'Premise'
      print(f'{i+1}. {step.formula} ({rule_name}, premises: {premises})')


def deduce(
  premises: t.List[Formula], conclusion: Formula, max_steps: int = 100
) -> t.Optional[Proof]:
  proof = Proof()

  for premise in premises:
    proof.add_step(premise, None, [])

  rules = [ModusPonens(), AndIntroduction(), AndElimination(), OrIntroduction()]

  for _ in range(max_steps):
    for rule in rules:
      for indices in product(range(len(proof.steps)), repeat=2):
        premises = [proof.steps[i].formula for i in indices]
        if rule.check(premises):
          result = rule.apply(premises)
          if result == conclusion:
            proof.add_step(conclusion, rule, list(indices))
            return proof
          elif result is not None:
            proof.add_step(result, rule, list(indices))

  return None


def verify_proof(
  proof: Proof, premises: t.List[Formula], conclusion: Formula
) -> bool:
  """
  Verifies if a given proof is valid and proves the conclusion from the premises.

  Args:
    proof (Proof): The proof to verify.
    premises (List[Formula]): The initial premises.
    conclusion (Formula): The conclusion to be proved.

  Returns:
    bool: True if the proof is valid and proves the conclusion, False otherwise.
  """
  if not proof.check_validity():
    return False

  proof_premises = [step.formula for step in proof.steps if step.rule is None]

  # Check if all premises are in the proof's premises
  if not all(premise in proof_premises for premise in premises):
    return False

  # Check if all proof's premises are in the given premises
  if not all(proof_premise in premises for proof_premise in proof_premises):
    return False

  return proof.steps[-1].formula == conclusion


def check_semantics(
  proof: Proof, premises: t.List[Formula], conclusion: Formula
) -> bool:
  """
  Checks if the proof is semantically valid using truth tables.

  Args:
      proof (Proof): The proof to check.
      premises (List[Formula]): The initial premises.
      conclusion (Formula): The conclusion of the proof.

  Returns:
      bool: True if the proof is semantically valid, False otherwise.
  """
  return is_semantic_consequence(premises, conclusion)


def parse_formula(formula_str: str) -> Formula:
  """
  Parses a string representation of a formula into a Formula object.

  Args:
    formula_str (str): The string representation of the formula.

  Returns:
    Formula: The parsed Formula object.

  Raises:
    ValueError: If the input string cannot be parsed into a valid Formula.
  """
  formula_str = formula_str.strip()

  if formula_str.startswith('(') and formula_str.endswith(')'):
    formula_str = formula_str[1:-1]

  if '∧' in formula_str:
    left, right = formula_str.split('∧')
    return Expression(Operator.AND, (parse_formula(left), parse_formula(right)))
  elif '∨' in formula_str:
    left, right = formula_str.split('∨')
    return Expression(Operator.OR, (parse_formula(left), parse_formula(right)))
  elif '→' in formula_str:
    left, right = formula_str.split('→')
    return Expression(
      Operator.IMPLIES, (parse_formula(left), parse_formula(right))
    )
  elif formula_str.startswith('¬'):
    return Expression(Operator.NOT, (parse_formula(formula_str[1:]),))
  else:
    return Proposition(formula_str)


# Example usage
if __name__ == '__main__':

  def construct_proof_interactively():
    proof = Proof()

    rules = {
      'mp': ModusPonens(),
      'and_intro': AndIntroduction(),
      'and_elim': AndElimination(),
      'or_intro': OrIntroduction(),
    }

    while True:
      formula = input("Enter formula (or 'done' to finish): ")

      if formula.lower() == 'done':
        break

      rule_name = input('Enter rule name: ')

      premises = list(
        map(
          int,
          input('Enter premise step numbers (comma-separated): ').split(','),
        )
      )

      formula_obj = parse_formula(formula)

      rule = rules.get(rule_name)

      if formula_obj and rule:
        proof.add_step(formula_obj, rule, premises)
      else:
        print('Invalid input. Please try again.')

    return proof

  def automated_proving(premises_str: t.List[str], conclusion_str: str):
    premises = [parse_formula(p) for p in premises_str]

    conclusion = parse_formula(conclusion_str)

    proof = deduce(premises, conclusion)

    if proof:
      print('Proof found:')
      proof.display()
    else:
      print('No proof found within the step limit.')

  def semantic_analysis(formula_str: str):
    formula = parse_formula(formula_str)
    print('Truth Table:')
    print(generate_truth_table(formula))

  # Define some propositions
  p = Proposition('p')
  q = Proposition('q')

  # Define a simple rule: Modus Ponens
  premises = [Expression(Operator.IMPLIES, (p, q)), p]
  conclusion = q

  # Try to find a proof
  proof = deduce(premises, conclusion)

  if proof:
    print('Proof found:')
    proof.display()
    print(f'Proof is valid: {verify_proof(proof, premises, conclusion)}')
    print(f'Semantically valid: {check_semantics(proof, premises, conclusion)}')
  else:
    print('No proof found.')

  # Demonstrate semantic analysis
  complex_formula = Expression(
    Operator.IMPLIES,
    (Expression(Operator.AND, (p, q)), Expression(Operator.OR, (p, q))),
  )
  print('\nTruth Table for (p ∧ q) → (p ∨ q):')
  print(generate_truth_table(complex_formula))

  # Interactive proof construction
  print('\nInteractive Proof Construction:')
  interactive_proof = construct_proof_interactively()
  print('\nConstructed Proof:')
  interactive_proof.display()

  # Automated proving
  print('\nAutomated Proving:')
  automated_proving(['p', 'p → q'], 'q')

  # Semantic analysis
  print('\nSemantic Analysis:')
  semantic_analysis('(p ∧ q) → (p ∨ q)')
