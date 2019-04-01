# PythonProlog

This is the work in progress version (branch prepareAI)


This version can proof infinity of prime numbers from a set of rules without intermediate handwritten proof steps.

it is trained in the prepareAI directory by 

First run python3 ../main.py to generate the training data and than python3 prepare.py --lr 0.0001,
model copied to final_eqn0.hdf5
and started python3 main.py

The search space seems to be drastically reduced, but it is not guarantied, that the proof succeeds, 
as only the first three guesses of the rnn are used.


ToDo:

- there seems to be a bug with recording track_for_ai, backtracking leaving one:
 ('predict2', 1, '[[mmod,[mult,[p,gproof],new],xproof],_0,isdeb]'), ('is', '[_0,[rnn,[mmod,[mult,[p,gproof],new],xproof],5]]'), ('predict2', 1, '[[mmod,[mult,[p,gproof],new],xproof],_0,isdeb]'), ('is', '[_0,[rnn,[mmod,[mult,[p,gproof],new],xproof],5]]'), ('predict2', 1, '[[mmod,[mult,[p,gproof],new],xproof],_0,isdeb]'), 

rnn support:
at the moment is(X,[rnn,term,mode]) and rnn(X,[term,mode]) is supported

rnn(X,[term, limit_number, limit_percent, DEB])

X: number of predicate to use
term: term to be interpreted by the rnn
limit_number: number of best results to be returned, if 0 all are returned
limit_persent: minimum percent the neural net must give for a predicate, to be returned
DEB: optionla debugging var, returning the order of the return values



Prolog interpreter written in Python.

It is planned to be used as a basis for a machine learning approach to learn in which order
the predicates should be executed. This way the progammer might, as the original idea of prolog, just
concentrate on the rules.

The test.pl file shows some usage and is executed before starting the prolog console.

```
Starting: (you might have to install python modules from requirements.txt)

python main.py 

[all_tests_are_OK]


#quit to leave prolog, #clear to clean database #load to load a file (adding ending .pl)
atom( .... ) format is only allowed for predicates, if you want to have add mult use lists with [add,X,Y]
also for structures use [ .....  ], internally it is transfered to [x0,x1] -> l(x0,l(x1,None))

PyProlog==> 


Usage:
Have a look at the test.pl file, there are some examples of allowed inputs. It is run before presenting the PyProlog==> promt,
which should result in [all_tests_are_OK].

PyProlog==> is(X,[add,4,5])?
[9,[add,4,5]]

PyProlog==> repeat(2)?
[2]
[2]

PyProlog==> 


Buildins:
cut(0)   is prolog cut ( ! in prolog syntax)
repeat(X) is repeat infinitly if X=0 otherwize repeat X times
is(X, expression) is the calculation build in. All calculations have a name and two operands
and are written as a list: [name, op1, op2]

PyProlog==> is(X,[add,3,[mult,2,5]])?
[13,[add,3,[mult,2,5]]]

supported operation names (all operands are integers!): 
add (op1 + op2)
mult(op1 * op2)
div (op1 / op2)
mod (op1 mod op2)
rand (return random op1 <= x <= op2)
lower (op1 < op2) return 1 if true, otherwize fails
lowereq (op1 <= op2)
neq (op1 != op2)
eq (op1 == op2)
```