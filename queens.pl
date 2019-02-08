queens(N,Qs):-range(1,N,Ns), queens3(Ns,[],Qs).
queens3([],Qs,Qs).
queens3(UnplacedQs,SafeQs,Qs):-select(UnplacedQs,UnplacedQs1,Q), not_attack(SafeQs,Q),queens3(UnplacedQs1,[Q|SafeQs],Qs).
not_attack(Xs,X):-not_attack3(Xs,X,1).
not_attack3([],X,Y):-cut(1).
not_attack3([Y|Ys],X,N):-is(Z1,[neq,X,[add,Y,N]]), is(Z2,[neq,X,[sub,Y,N]]),is(N1,[add,N,1]),not_attack3(Ys,X,N1).
select([X|Xs],Xs,X).
select([Y|Ys],[Y|Zs],X):-select(Ys,Zs,X).
range(N,N,[N]):-cut(1).
range(M,N,[M|Ns]):-is(Z1,[lower,M,N]),is(M1,[add,M,1]),range(M1,N,Ns).
queens(8,X)?