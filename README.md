# PythonProlog

This is the work in progress version (branch prepareAI)

ToDo:

- not clear, how to allow broad search with it? But maybe it must not be handled by the engine immediatly, but can be done by code as 

eqn1([mult,X,Y],[mult,X1,Y1],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), eqn1(X,X1,C1,DD), eqn1(Y,Y1,C1,DD).
eqn1(X,Y,CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]),eqn1(X,Z,C1,DD),eqn1(Z,Y,C1,DD).



rnn support:
at the moment is(X,[rnn,term,mode]) and rnn(X,[term,mode]) is supported

the rnn allowes mode order, which returns all...





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