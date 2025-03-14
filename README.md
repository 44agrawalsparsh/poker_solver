# poker_solver
Terminal tooling to practice games against GTO agents

Full credit to the solver engine: https://github.com/b-inary/postflop-solver

Notes on usage:

Preflop ranges are up to the user, system expects format such as

ranges/CO/2.5bb/SB/8bb/BB/call/CO/call/CO.txt to mean CO's range when action folds to them, they raise to 2.5bb, action folds to SB, they raise to 8bb, BB calls then action folds to CO who calls

likewise ranges/CO/2.5bb/SB/8bb/BB/call/CO/call/BB.txt would be the BB's range in this position.

run generate_preflop_strategy.py to process these ranges into a strategy that the application can use.

Afterwards, simply call play.py

This is a work in progress, but should still be very useful for practice!

Will try to add documentation in the near future!

![Demo](https://github.com/agrawalsparsh/poker_solver/blob/main/figures/output.gif)
