# PythonProlog
Prolog interpreter written in Python.

It is planned to be used as a basis for a machine learning approach to learn in which order
the predicates should be executed. This way the progammer might, as the original idea of prolog, just
concentrate on the rules.

The test.pl file shows some usage and is executed before starting the prolog console.


This is the work in progress version (branch prepareAI)

ToDo:

- introduce a prolog buildin to question the NN to prefer a rule?
- not clear, how to allow broad search with it? But maybe it must not be handled by the engine immediatly, but can be done by code as 

eqn1([mult,X,Y],[mult,X1,Y1],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), eqn1(X,X1,C1,DD), eqn1(Y,Y1,C1,DD).
eqn1(X,Y,CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]),eqn1(X,Z,C1,DD),eqn1(Z,Y,C1,DD).

