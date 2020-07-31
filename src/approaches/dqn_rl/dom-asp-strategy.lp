% #maximize {N1+N2:does(_,plays(domino(N1,N2), _))}.
amount_of_n(P,N,M):- #count{1,L,N: true(in_hand(P, domino(L,N)));1,R,N: true(in_hand(P, domino(N,R)))}=M,role(P),eye(N).
#minimize{M@2,S: next(stack(S,N)), amount_of_n(P,N,M), next(control(P))}.
% #maximize{M@1,S: next(stack(S,N)), amount_of_n(P,N,M), true(control(P))}.
