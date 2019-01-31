eqq(X,X).
eqn0([p,empty], ee).
eqn0([p,[g,X,G]],[mult,X,[p,G]]).
eqn0([mult,A, nn],A).
eqn0([mult,[mult,A,B],C],[mult,A,[mult,B,C]]).
eqn0([mult,A,[add,B,C]],[add,[mult,A,B],[mult,A,C]]).
eqn0([add,A,nn],A).
eqn0([add,A,ee],[s,A]).
eqn0([add,A,[s,B]],[s,[add,A,B]]).
eqn0([mmod,ee,[s,X]],ee).
eqn0([mmod,[add,A,B],C],[mmod,[add,[mmod,A,C],[mmod,B,C]],C]).
eqn0([mmod,N,N],ee).
eqn0(X,X).
eqn1(X,Y,CC,DD):-eqn0(X,Y).
eqn1(X,Y,CC,DD):-eqn0(Y,X).
eqn1([mmod,X,Y],[mmod,X1,Y1],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), eqn1(X,X1,C1,DD), eqn1(Y,Y1,C1,DD).
eqn1([add,X,Y],[add,X1,Y1],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), eqn1(X,X1,C1,DD), eqn1(Y,Y1,C1,DD).
eqn1([mult,X,Y],[mult,X1,Y1],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), eqn1(X,X1,C1,DD), eqn1(Y,Y1,C1,DD).
eqn1(X,Y,CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]),eqn0(X,Z),eqn1(Z,Y,C1,DD).
eqn2(X,Y,CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]),eqn0(Z,X),eqn1(Z,Y,C1,DD).
eqn2(ee,Y,1,2)?