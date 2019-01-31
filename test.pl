peter(X,auto).
peter(Y,wasser).
cuttest(X) :- peter(X,X), cut(1), peter(X,X).
cuttest(X)?
calculate(X,Y):-is(Y,[add,X,[mult,X,2]]).
eqq(X,X).
test3([X,Y]):-eqq(X,Y).
fak(1,1):-cut(1).
fak(X,Y):-is(X1,[sub,X,1]),fak(X1,Y1),is(Y,[mult,Y1,X]).
fak(4,24)?
dotest0(X,Y,Z,ZZ):- cuttest(X),test3([Y,4]),calculate(3,Z),fak(4,24),eqq(ZZ,all_tests_are_OK).
dotest0(X,Y,Z,testsFailed).
dotest(ZZ):-dotest0(wasser,4,9,ZZ),cut(1),is(X,[write,tttt,track]).
dotest(ZZ)?
