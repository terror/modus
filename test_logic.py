from itertools import product

import pytest

from logic import (
  Expression,
  Operator,
  Proposition,
  evaluate,
  generate_truth_table,
  get_variables,
  is_graph_colorable,
  is_semantic_consequence,
)


@pytest.fixture
def propositions():
  return Proposition('p'), Proposition('q'), Proposition('r')


def test_proposition_evaluation():
  assert evaluate(Proposition(True))
  assert not evaluate(Proposition(False))
  assert not evaluate(Proposition('p'))
  assert evaluate(Proposition('p'), {'p': True})


def test_and_operator(propositions):
  p, q, _ = propositions
  expr = Expression(Operator.AND, (p, q))
  assert not evaluate(expr, {'p': True, 'q': False})
  assert evaluate(expr, {'p': True, 'q': True})


def test_or_operator(propositions):
  p, q, _ = propositions
  expr = Expression(Operator.OR, (p, q))
  assert evaluate(expr, {'p': True, 'q': False})
  assert not evaluate(expr, {'p': False, 'q': False})


def test_not_operator(propositions):
  p, _, _ = propositions
  expr = Expression(Operator.NOT, (p,))
  assert evaluate(expr, {'p': False})
  assert not evaluate(expr, {'p': True})


@pytest.mark.parametrize(
  'p, q, expected',
  [
    (False, False, True),
    (False, True, True),
    (True, False, False),
    (True, True, True),
  ],
)
def test_implies_operator(propositions, p, q, expected):
  p_prop, q_prop, _ = propositions
  expr = Expression(Operator.IMPLIES, (p_prop, q_prop))
  assert evaluate(expr, {'p': p, 'q': q}) == expected


def test_complex_expression(propositions):
  p, q, r = propositions

  expr = Expression(
    Operator.OR,
    (
      Expression(Operator.AND, (p, q)),
      Expression(Operator.IMPLIES, (q, r)),
    ),
  )

  assert evaluate(expr, {'p': True, 'q': True, 'r': False})

  assert not evaluate(expr, {'p': False, 'q': True, 'r': False})


def test_get_variables(propositions):
  p, q, r = propositions
  expr = Expression(
    Operator.OR,
    (
      Expression(Operator.AND, (p, q)),
      Expression(Operator.IMPLIES, (q, r)),
    ),
  )
  assert get_variables(expr) == {'p', 'q', 'r'}


def test_is_semantic_consequence(propositions):
  p, q, _ = propositions
  premise1 = Expression(Operator.IMPLIES, (p, q))
  premise2 = p
  conclusion = q
  assert is_semantic_consequence([premise1, premise2], conclusion)

  premise3 = Expression(Operator.OR, (p, q))
  conclusion2 = Expression(Operator.AND, (p, q))
  assert not is_semantic_consequence([premise3], conclusion2)


def test_compactness_theorem():
  def generate_formula(n):
    return Expression(
      Operator.IMPLIES, (Proposition(f'p{n}'), Proposition(f'p{n+1}'))
    )

  def is_satisfiable(formulas):
    variables = set.union(*(get_variables(f) for f in formulas))
    return any(
      all(evaluate(f, dict(zip(variables, values))) for f in formulas)
      for values in product([True, False], repeat=len(variables))
    )

  # Test finite subsets
  for n in range(1, 10):
    subset = [Proposition('p1')] + [generate_formula(i) for i in range(1, n)]
    assert is_satisfiable(subset), f'Subset of size {n} should be satisfiable'

  # Demonstrate infinite case
  large_subset = [Proposition('p1')] + [
    generate_formula(i) for i in range(1, 1000)
  ]

  all_true = {'p' + str(i): True for i in range(1, 1001)}

  any_false = {'p' + str(i): i != 1 for i in range(1, 1001)}

  assert all(
    evaluate(f, all_true) for f in large_subset
  ), 'All formulas should be true when all pi are true'

  assert not all(
    evaluate(f, any_false) for f in large_subset
  ), 'Not all formulas can be true if any pi is false'


def test_truth_table_with_intermediate_results(propositions):
  p, q, r = propositions

  expr = Expression(Operator.IMPLIES, (Expression(Operator.IMPLIES, (p, q)), r))

  table = generate_truth_table(expr)

  lines = table.strip().split('\n')

  assert all(x in lines[1] for x in ['p', 'q', 'r', '(p → q)', '((p → q) → r)'])

  content_lines = [line for line in lines[3:] if '┌' not in line]

  expected_content = [
    ['0', '0', '0', '0', '0', '1', '0', '0'],
    ['0', '0', '1', '0', '0', '1', '1', '1'],
    ['0', '1', '0', '0', '1', '1', '0', '0'],
    ['0', '1', '1', '0', '1', '1', '1', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '1'],
    ['1', '0', '1', '1', '0', '0', '1', '1'],
    ['1', '1', '0', '1', '1', '1', '0', '0'],
    ['1', '1', '1', '1', '1', '1', '1', '1'],
  ]

  for expected, actual in zip(expected_content, content_lines):
    assert expected == [
      cell.strip() for cell in actual.split('│') if cell.strip()
    ]


def test_repr(propositions):
  p, q, r = propositions
  assert repr(Expression(Operator.AND, (p, q))) == '(p ∧ q)'
  assert repr(Expression(Operator.OR, (p, q))) == '(p ∨ q)'
  assert repr(Expression(Operator.NOT, (p,))) == '¬p'
  assert repr(Expression(Operator.IMPLIES, (p, q))) == '(p → q)'
  assert (
    repr(Expression(Operator.IMPLIES, (Expression(Operator.AND, (p, q)), r)))
    == '((p ∧ q) → r)'
  )


@pytest.mark.parametrize(
  'graph, num_colors, expected',
  [
    ({'A': ['B', 'C'], 'B': ['A', 'C'], 'C': ['A', 'B']}, 2, False),
    ({'A': ['B', 'C'], 'B': ['A', 'C'], 'C': ['A', 'B']}, 3, True),
    (
      {'A': ['B', 'D'], 'B': ['A', 'C'], 'C': ['B', 'D'], 'D': ['A', 'C']},
      2,
      True,
    ),
    (
      {
        'A': ['B', 'C', 'D'],
        'B': ['A', 'C', 'D'],
        'C': ['A', 'B', 'D'],
        'D': ['A', 'B', 'C'],
      },
      3,
      False,
    ),
    (
      {
        'A': ['B', 'C', 'D'],
        'B': ['A', 'C', 'D'],
        'C': ['A', 'B', 'D'],
        'D': ['A', 'B', 'C'],
      },
      4,
      True,
    ),
  ],
)
def test_graph_coloring(graph, num_colors, expected):
  assert is_graph_colorable(graph, num_colors) == expected
