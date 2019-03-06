eqq(X,X).

eqn0([p,empty], ee).
eqn0([p,[g,X,G]],[mult,X,[p,G]]).
eqn0([mult,A, ee],A).
eqn0([mult,A, nn],nn).
eqn0([mult,[mult,A,B],C],[mult,A,[mult,B,C]]).
eqn0([mult,A,[add,B,C]],[add,[mult,A,B],[mult,A,C]]).
eqn0([add,A,nn],A).
eqn0([add,A,ee],[s,A]).
eqn0([add,A,[s,B]],[s,[add,A,B]]).
eqn0([mmod,ee,[s,X]],ee).
eqn0([mmod,[add,A,B],C],[mmod,[add,[mmod,A,C],[mmod,B,C]],C]).
eqn0([mmod,[mult,A,B],C],[mmod,[mult,[mmod,A,C],[mmod,B,C]],C]).
eqn0([mmod,A,A],nn).
eqn0([mmod,nn,A],nn).
eqn0([mmod,[mult,N,X],N],nn).
eqn0([mult,A,B],[mult,B,A]).

try1(X):-repeat(4),is(X,[rand,0,8]).
try(X).

eqn1(0,X,Y,CC,DD):-eqn0(X,Y).
eqn1(1,X,Y,CC,DD):-eqn0(Y,X).
eqn1(2,[mmod,X,Y],[mmod,X1,Y1],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), try(T1), eqn1(T1,X,X1,C1,DD), eqq(Y,Y1).
eqn1(3,[mmod,X,Y],[mmod,X1,Y1],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), eqq(X,X1), try(T2), eqn1(T2,Y,Y1,C1,DD).
eqn1(4,[add,X,Y],[add,X1,Y1],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), try(T1), eqn1(T1,X,X1,C1,DD), eqq(Y,Y1).
eqn1(5,[add,X,Y],[add,X1,Y1],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), eqq(X,X1), try(T2), eqn1(T2,Y,Y1,C1,DD).
eqn1(6,[mult,X,Y],[mult,X1,Y1],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), try(T1), eqn1(T1,X,X1,C1,DD), eqq(Y,Y1).
eqn1(7,[mult,X,Y],[mult,X1,Y1],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), eqq(X,X1), try(T2), eqn1(T2,Y,Y1,C1,DD).
eqn1(8,X,Y,CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), try(T1), eqn1(T1,X,Z,C1,DD), try(T2), eqn1(T2,Z,Y,C1,DD).

eqn0([mmod,[p,gproof],xproof],nn).

eqn2(X,Y,CC,DD,F1,F2):-eqn1(T,X,Y,CC,DD),is(RR,[write,F1,Y]),is(RRR,[write,F2,track]).
eqn2(ee,Y,1,1,tttx1,tttt1)?
eqn2(nn,Y,1,1,tttx1,tttt1)?
eqn2(ee,Y,1,2,tttx2,tttt2)?
eqn2(nn,Y,1,2,tttx2,tttt2)?
eqn2(ee,Y,1,3,tttx3,tttt3)?
eqn2(nn,Y,1,3,tttx3,tttt3)?
eqn2(ee,Y,1,4,tttx4,tttt4)?
eqn2(nn,Y,1,4,tttx4,tttt4)?

