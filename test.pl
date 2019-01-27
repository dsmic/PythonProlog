peter(X,auto).
peter(Y,wasser).
peter(X,Y)?
cuttest(X) :- peter(X,X), cut(1).
is(A,[mult,3,3])?
cuttest(X)?
calc(X,Y):-is(Y,[add,X,[mult,X,2]]).
calc(3,C)?