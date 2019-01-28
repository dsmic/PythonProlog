peter(X,auto).
peter(Y,wasser).
peter(X,Y)?
cuttest(X) :- peter(X,X), cut(1), peter(X,X).
is(A,[mult,3,3])?
cuttest(X)?
calculate(X,Y):-is(Y,[add,X,[mult,X,2]]).
calculate(3,C)?
eqq(X,X).
test3([X,Y]):-eqq(X,Y).
test3([W,4])?
dotest0(X,Y,Z,ZZ):- cuttest(X),test3([Y,4]),calculate(3,Z),eqq(ZZ,all_tests_are_OK).
dotest0(fail,fail,fail,testsFailed).
dotest(ZZ):-dotest0(auto,4,9,ZZ),cut(1).
dotest(ZZ)?