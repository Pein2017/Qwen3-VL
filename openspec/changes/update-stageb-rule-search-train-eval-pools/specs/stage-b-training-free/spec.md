## MODIFIED Requirements

### Rule-Search Pools and Naming
#### Scenario: Train/Eval pool semantics
- WHEN rule_search is configured,
- THEN it SHALL distinguish **train pools** (proposal + gate) from **eval pools** (monitor/veto only),
- AND config naming SHALL reflect train vs eval usage unambiguously.

#### Scenario: Fixed eval pool
- WHEN a run starts,
- THEN the eval pool SHALL be sampled once and remain fixed for the run,
- AND eval tickets SHALL NOT participate in proposal generation or train-gate evaluation.

#### Scenario: Rolling train pool
- WHEN iterating rule_search,
- THEN the train pool SHALL advance across the remaining tickets in fixed-size batches,
- AND MAY wrap/reshuffle after exhaustion when configured.

#### Scenario: Eval veto
- WHEN a candidate passes the train gate,
- THEN the candidate SHALL be rejected if eval metrics degrade beyond a configured threshold,
- AND the veto threshold SHALL be configurable (default: max acc drop 0.01 absolute accuracy drop).
