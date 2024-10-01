import pytest

from modus import (
  Expression,
  ModusPonens,
  Operator,
  Proof,
  Proposition,
  check_semantics,
  deduce,
  evaluate,
  generate_truth_table,
  get_variables,
  is_graph_colorable,
  is_semantic_consequence,
  parse_formula,
  verify_proof,
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


def test_truth_table_generation(propositions):
  p, q, _ = propositions
  expr = Expression(Operator.IMPLIES, (p, q))
  table = generate_truth_table(expr)
  assert 'p' in table
  assert 'q' in table
  assert '(p → q)' in table


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


def test_modus_ponens():
  p = Proposition('p')
  q = Proposition('q')

  premise1 = Expression(Operator.IMPLIES, (p, q))
  premise2 = p
  conclusion = q

  proof = deduce([premise1, premise2], conclusion)

  assert proof is not None
  assert verify_proof(proof, [premise1, premise2], conclusion)
  assert check_semantics(proof, [premise1, premise2], conclusion)


# def test_and_introduction():
#   p = Proposition('p')
#   q = Proposition('q')

#   conclusion = Expression(Operator.AND, (p, q))

#   proof = deduce([p, q], conclusion)

#   assert proof is not None
#   assert verify_proof(proof, [p, q], conclusion)
#   assert check_semantics(proof, [p, q], conclusion)


# def test_and_elimination():
#   p = Proposition('p')
#   q = Proposition('q')

#   premise = Expression(Operator.AND, (p, q))

#   proof1 = deduce([premise], p)

#   assert proof1 is not None
#   assert verify_proof(proof1, [premise], p)
#   assert check_semantics(proof1, [premise], p)

#   proof2 = deduce([premise], q)

#   assert proof2 is not None
#   assert verify_proof(proof2, [premise], q)
#   assert check_semantics(proof2, [premise], q)


# def test_or_introduction():
#   p = Proposition('p')
#   q = Proposition('q')
#   conclusion = Expression(Operator.OR, (p, q))

#   proof1 = deduce([p], conclusion)
#   assert proof1 is not None
#   assert verify_proof(proof1, [p], conclusion)
#   assert check_semantics(proof1, [p], conclusion)

#   proof2 = deduce([q], conclusion)
#   assert proof2 is not None
#   assert verify_proof(proof2, [q], conclusion)
#   assert check_semantics(proof2, [q], conclusion)


def test_parse_formula():
  assert parse_formula('p') == Proposition('p')

  assert parse_formula('(p ∧ q)') == Expression(
    Operator.AND, (Proposition('p'), Proposition('q'))
  )

  assert parse_formula('(p ∨ q)') == Expression(
    Operator.OR, (Proposition('p'), Proposition('q'))
  )

  assert parse_formula('(p → q)') == Expression(
    Operator.IMPLIES, (Proposition('p'), Proposition('q'))
  )

  assert parse_formula('¬p') == Expression(Operator.NOT, (Proposition('p'),))


def test_proof_validity():
  p = Proposition('p')
  q = Proposition('q')

  premise = Expression(Operator.IMPLIES, (p, q))

  valid_proof = Proof()
  valid_proof.add_step(premise, None, [])
  valid_proof.add_step(p, None, [])
  valid_proof.add_step(q, ModusPonens(), [0, 1])

  assert verify_proof(valid_proof, [premise, p], q)

  invalid_proof = Proof()
  invalid_proof.add_step(premise, None, [])
  invalid_proof.add_step(p, None, [])
  invalid_proof.add_step(q, None, [])  # Invalid step

  assert not verify_proof(invalid_proof, [premise, p], q)


if __name__ == '__main__':
  pytest.main()
