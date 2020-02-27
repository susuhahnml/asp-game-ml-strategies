#constant(cell,1).
#constant(cell,2).
#constant(cell,3).
#constant(player,a).
#constant(player,b).
#modeo(4,next(has(var(player),var(cell))),(positive)).
#modeo(1,in_line(var(cell),var(cell),var(cell)),(positive)).
#modeo(3,next(free(var(cell))),(positive)).
#modeo(1,true(control(var(player))),(positive)).
#modeo(1,next(control(var(player))),(positive)).
#weight(-1).
#weight(1).
#maxp(2).
#maxv(5).


% Expected strategy

% :~ in_line(V1,V2,V3), next(has(P,V1)), next(has(P,V2)), next(has(P,V3)), true(control(P)).[-2@2]
% :~ in_line(V1,V2,V3), next(has(P,V1)), next(has(P,V2)), next(free(V3)), next(control(P)).[1@1]
% :~ in_line(V1,V2,V3), next(has(P,V1)), next(has(P,V3)), next(free(V2)), next(control(P)).[1@1]
% :~ in_line(V1,V2,V3), next(has(P,V3)), next(has(P,V2)), next(free(V1)), next(control(P)).[1@1]
