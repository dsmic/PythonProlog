peter(X,auto).
peter(Y,wasser).
cuttest(X) :- peter(X,X), cut(1), peter(X,X).
cuttest(X)?
calculate(X,Y):-is(Y,[add,X,[mult,X,2]]).
eqq(X,X).
test3([X,Y]):-eqq(X,Y).
<<<<<<< HEAD
fak(1,1):-cut(1).
fak(X,Y):-is(X1,[sub,X,1]),fak(X1,Y1),is(Y,[mult,Y1,X]).
fak(4,24)?
dotest0(X,Y,Z,ZZ):- cuttest(X),test3([Y,4]),calculate(3,Z),fak(4,24),eqq(ZZ,all_tests_are_OK).
dotest0(X,Y,Z,testsFailed).
dotest(ZZ):-dotest0(wasser,4,9,ZZ),cut(1).
tt(X):-repeat(3),is(X,[rand,1,4]).
tt(X)?
dotest(ZZ)?
=======
test3([W,4])?
dotest0(X,Y,Z,ZZ):- cuttest(X),test3([Y,4]),calc(3,Z),eqq(ZZ,all_tests_are_OK).
dotest0(fail,fail,fail,testsFailed).
dotest(ZZ):-dotest0(auto,4,9,ZZ),cut(1).
fak(1,1).
fak(X,Y):-is(X1,[sub,X,1]),fak(X1,Y1),is(Y,[mult,X,Y1]).
fak2(X,Y):-fak(X,Y),cut(1).
fak2(4,S)?
dotest(ZZ)?
>>>>>>> 85d9016... - some documentation in readme
