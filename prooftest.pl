eqq(X,X).

eqn0(0,[p,empty], ee).
eqn0(1,[p,[g,X,G]],[mult,X,[p,G]]).
eqn0(2,[mult,A, ee],A).
eqn0(3,[mult,A, nn],nn).
eqn0(4,[mult,[mult,A,B],C],[mult,A,[mult,B,C]]).
eqn0(5,[mult,A,[add,B,C]],[add,[mult,A,B],[mult,A,C]]).
eqn0(6,[add,A,nn],A).
eqn0(7,[add,A,ee],[s,A]).
eqn0(8,[add,A,[s,B]],[s,[add,A,B]]).
eqn0(9,[mmod,ee,[s,X]],ee).
eqn0(10,[mmod,[add,A,B],C],[mmod,[add,[mmod,A,C],[mmod,B,C]],C]).
eqn0(11,[mmod,[mult,A,B],C],[mmod,[mult,[mmod,A,C],[mmod,B,C]],C]).
eqn0(12,[mmod,A,A],nn).
eqn0(13,[mmod,nn,A],nn).
eqn0(14,[mmod,[mult,N,X],N],nn).
eqn0(15,[mult,A,B],[mult,B,A]).

eqn0(16,[mmod,[p,gproof],xproof],nn).

predictdeb(X,Y):-repeat(3), is(Y,[rnn,X,30]), is(ZZ,[write,predict,[predict,X,Y]]).
predict1(X,Y):-repeat(10), is(Y,[rnn,X,10]).
predict1(X,Y1):-is(Y,[rnn,X,best]),is(ZZ,[write,predict,[predict,X,Y]]).
predict1(X,Y):-rnn(Y,[X,3,0]),is(ZZ,[write,predict,[predict,X,Y]]).

predict22(X,Y, DEB):-rnn(Y,[X,2,0,DEB]).
predict2(X,Y,isdeb):-is(Y,[rnn,X,0]).
predict22(X,Y,isdeb):-is(Y,[rand,0,16]).

predict22(X,Y,debug).

predict(X,Y,debug).

output(X,Y):-is(ZZ,[write,predict,[predict,X,Y]]).
output1(X,Y).

eqn1(1,X,Y,CC,DD):-predict2(X,A,DEB), eqn0(A,X,Y).
eqn1(0,X,Y,CC,DD):-predict2(Y,A,DEB), eqn0(A,Y,X).
eqn1(2,[mmod,X,Y],[mmod,X1,Y],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), predict(X,O0,DEB), eqn1(O0,X,X1,C1,DD), output(O0, DEB).
eqn1(3,[mmod,X,Y],[mmod,X,Y1],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), predict(Y,O1,DEB2), eqn1(O1,Y,Y1,C1,DD), output(O1, DEB2).
eqn1(4,[add,X,Y],[add,X1,Y],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), predict(X,O0,DEB), eqn1(O0,X,X1,C1,DD), output(O0, DEB).
eqn1(5,[add,X,Y],[add,X,Y1],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), predict(Y,O1,DEB2), eqn1(O1,Y,Y1,C1,DD), output(O1, DEB2).
eqn1(6,[mult,X,Y],[mult,X1,Y],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), predict(X,O0,DEB), eqn1(O0,X,X1,C1,DD), output(O0, DEB).
eqn1(7,[mult,X,Y],[mult,X,Y1],CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), predict(Y,O1,DEB2), eqn1(O1,Y,Y1,C1,DD), output(O1, DEB2).
eqn1(8,X,Y,CC,DD):-is(Z1,[lower,CC,DD]), is(C1,[add,CC,1]), predict(X,O0,DEB),eqn1(O0,X,Z,C1,DD), output(O0, DEB), predict(Z,O1,DEB2), eqn1(O1,Z,Y,C1,DD), output(O1, DEB2).

eqn111(6,X,X,CC,DD):-is(Z1,[lower,CC,DD]).



eqn1(O,[mmod,[mult,[p,gproof],new],xproof], [mmod,[mult,[mmod,[p,gproof],xproof],[mmod,new,xproof]],xproof],1,1)?
eqn1(O,[mmod,[mult,[mmod,[p,gproof],xproof],[mmod,new,xproof]],xproof], [mmod,[mult,nn,[mmod,new,xproof]],xproof],1,4)?
eqn1(O,[mmod,[mult,nn,[mmod,new,xproof]],xproof], nn,1,4)?

start(X):-repeat(0),eqn1(O,[mmod,[mult,[p,gproof],new],xproof], nn,1,5).
start(0)?
start(0)?
start(0)?
start(0)?
start(0)?
start(0)?
start(0)?
start(0)?
start(0)?
start(0)?
