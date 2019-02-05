-module(hw1).

-export([kth_largest/2]).  % for Q1
-export([count_make/0, count_make/1, count_proc/1, count_next/1, count_end/1]). % for Q2
-export([mean/1, esq/1, cov/1, cov/2]). % for Q3

-export([close/2, close/3, cov_check/2, cov_check/3, cov_data/4, cov_leaf/1]). % help with testing cov(W, Key)
-export([your_answer/2]). % throws an error if you try executing one of the stubs

% kth_largest(K, List) -> the K^th largest element of List.
% If List has duplicate elements, we count each of them when determining
% the Kth largest.  Examples:
%   hw1:kth_largest(3, [2,5,1,2,3,7,6]) -> 5
%   kth_largest(3, [2,8,3,7,8,4,6]) -> 7
kth_largest(K, List) when is_integer(K) andalso K > 0 andalso is_list(List) andalso length(List) >= K-> 
  Acc = lists:sort(lists:sublist(List, K)),
  NewL = trimK(K, List),
  [KthElem|_] = kth2(K, NewL, Acc),
  KthElem.

kth2(K, [H|T], Acc) -> % N times
  kth2(K, T, maintain(H, Acc));
kth2(_, [], Acc) -> 
  Acc.

maintain(K, [H|T]) when K < H -> % Smaller than first elem
  [H|T];
maintain(K, [H|T]) -> 
  maintain(K, [H], T).

maintain(K, Front, [H|T]) when K >= H -> % Should run less close to K times as you progress through the list...
  maintain(K, Front ++ [H], T);
maintain(K, [_|T], Back) -> % K =< H, insert. These ops should be constant time I hope
  T ++ [K|Back].

trimK(0, List) ->
  List;
trimK(K, [_|T]) ->
  trimK(K-1, T).



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
% Functions for Q2                                                         %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% count_make(InitialCount) -> CounterProcess
count_make(InitialCount) ->
  spawn(?MODULE, count_proc, [InitialCount]).

% count_make(): equivalent to count_make(0)
count_make() ->
  count_make(0).

% count_proc(CurrentCount) -> ok
%   The tail-recursive function for a counter-process.
%   N is the current count.
count_proc(CurrentCount) ->
  receive
    {count, ReplyTo} ->
      ReplyTo ! {count, self(), CurrentCount},
      count_proc(CurrentCount + 1);
    exit ->
      ok
  end.


% count_next(CounterPid)
%   Interface function for a counter process.
%   Send CountPid a 'next' message to increment the counter.
%   Receive the current (i.e. pre-increment) value of the counter and return it.
count_next(CounterPid) ->
  CounterPid ! {count, self()},
  receive
    {count, CounterPid, CurrentCount} ->
      CurrentCount
  end.

% count_next(CounterPid) -> ok.
%   Terminate a counter process.
count_end(CounterPid) ->
  CounterPid ! exit.

% You will probably want to write one or more functions that
%   create and use a counter process (or counter processes)
%   to determine the order in which a list comprehension evaluates
%  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
% Functions for Q3                                                         %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% mean(X) -> Mean
%   In my implementation, I allow X to be a list of numbers, or a nested
%   list (of nubmers).
%   If X is a list of numbers, e.g. [1, 3, 5, 7], then mean(X) is the
%     obvious mean, e.g. mean([1, 3, 5, 7]) -> 4.
%   If X is a list of lists, each element of X must have the same "shape".
%     In this case, we compute the element-wise mean of the elements of X.
%     For example if
%       X = [ [1,  2,   3],
%             [4,  8,  10],
%             [8, 12,  -4],
%             [2,  7,  18] ]
%     Then, mean(X) = [3.75, 7.25, 6.75].
mean(List) ->
  D = length(lists:nth(1, List)),
  [col_mean(List, ColNum) || ColNum <- lists:seq(1, D)].

col_mean(FullList, Col) ->
  NewList = [lists:nth(Col, X) || X <- FullList],
  lists:sum(NewList) / length(NewList).

% In my solution, I wrote a few functions for operations on vectors and
%   matrices that are useful for the rest of the problem.

% esq(X) -> E12
%   X is a list of vectors.
%     I.e. each element of X is a list of numbers, and all of these lists
%     must be of the same length.
%   E12 is an upper triangular matrix.
%   Let K be the length of each element of K.  Let 1 =< I =< J =< K.
%   Then lists:nth(J2, lists:nth(J1, E12)) is
%     (1/N)*sum_I=1^N lists:nth(J1, lists:nth(I, X)) * lists:nth(J2, lists:nth(I, X))
%   OTOTH, if you compute it that way, your big-O runtime will be too high because
%   lists:nth(N, List) takes time O(N).  You should write an efficient implemention.
esq(X) ->
  N = length(X),
  D = length(lists:nth(1, X)),
  J2s = lists:seq(1, D),
  [single_col(X, J2, N, D) || J2 <- J2s].

single_col(Matrix, J2, N, D) ->
  J1s = lists:seq(1, D),
  [single_place(Matrix, J1, J2, N) || J1 <- J1s].

single_place(Matrix, J1, J2, N) -> 
  lists:sum([mult(X, J1, J2) || X <- Matrix]) / N.

mult(X, J1, J2) ->
  lists:nth(J1, X) * lists:nth(J2, X).

% cov(X) -> COV
%   X is a list of vectors.
%     I.e. each element of X is a list of numbers, and all of these lists
%     must be of the same length.
%   COV is an upper triangular matrix.
%   Let K be the length of each element of K.  Let 1 =< I =< J =< K.
%   Then lists:nth(J2, lists:nth(J1, COV)) is our estimate of
%     E[(X_J1 - E[X_J1])(X_J2 - E[X_J2])].
cov(X) ->
  Means = mean(X),
  Esq = esq(X),
  J2s = lists:seq(1, length(X)),
  J1s = lists:seq(1, length(lists:nth(1, X))),
  [[cov_val(Esq, Means, J1, J2) || J1 <- J1s] || J2 <- J2s].

cov_val(Esq, Means, J1, J2) -> 
  M1 = lists:nth(J2, Means),
  M2 = lists:nth(J1, Means),
  lists:nth(J2, lists:nth(J1, Esq)) - M1 * M2.
  
% I wrote functions for cov_leaf and cov_combine to use
%   with wtree:reduce in cov(W, Key) below.

% cov(W, Key) -> COV
%   W is a worker tree (i.e. created using wtree:create).
%   Key is the key with which a distributed list of vectors is associated.
%   For example, let X be a list of 1000 vectors, where each vector has 5 elements.
%   If W has 10 worker processes, we could store 100 elements of X in the ProcState
%   of each worker process.  Let's assume that these pieces of X are associated
%   with the Key, 'data'.  Then, each process can get its piece of X with the expression
%     workers:get(ProcState, data)
%   Or, more generally,
%     workers:get(ProcState, Key)
%
%   cov(W, Key) should use wtree:reduce/3 or wtree:reduce/4 to compute the covariance
%   matrix for this data.  Just as with cov(X), cov(W, Key) should return an
%   upper triangular matrix.
cov(W, Key) ->
  wtree:reduce(W,
    fun(ProcState) -> cov_leaf(wtree:get(ProcState, Key)) end,  % Leaf
    fun(Left, Right) -> cov_combine(Left, Right) end, % Combine
    fun(Result) -> cov_root(Result) end
  ).

cov_leaf([]) -> none;

cov_leaf(Matrix) -> esq2(Matrix).

esq2(X) ->
  N = length(X),
  D = length(lists:nth(1, X)),
  J2s = lists:seq(1, D),
  Var_sums = [single_col2(X, J2, D) || J2 <- J2s],
  Mean_sums = vector_sum(X),
  {Mean_sums, Var_sums, N}.

single_col2(Matrix, J2, D) ->
  J1s = lists:seq(1, D),
  [single_place2(Matrix, J1, J2) || J1 <- J1s].

single_place2(Matrix, J1, J2) -> 
  lists:sum([mult2(X, J1, J2) || X <- Matrix]).

mult2(X, J1, J2) ->
  lists:nth(J1, X) * lists:nth(J2, X).

cov_root({Means, Vars, N}) -> 
  Final_means = lists:map(fun(X) -> X/N end, Means),
  ESQ = matrix_map(fun(X) -> X/N end, Vars),
  J2s = lists:seq(1, length(ESQ)),
  J1s = lists:seq(1, length(lists:nth(1, ESQ))),
  [[cov_val(ESQ, Final_means, J1, J2) || J1 <- J1s] || J2 <- J2s].

cov_combine(Left, Right) ->
  case {Left, Right} of
    {Left, none} -> Left;
    {none, Right} -> Right;
    {Left, Right} -> combine(Left, Right)
  end.

combine({Mean1, Var1, N1}, {Mean2, Var2, N2}) ->
  Mean_sums = vector_sum([Mean1, Mean2]),
  Var_sums = lists:zipwith(fun(X,Y) -> vector_sum([X,Y]) end, Var1, Var2),
  {Mean_sums, Var_sums, N1 + N2}.

vector_sum([]) ->
  [];

vector_sum([H|[]]) ->
  H;

vector_sum([H|T]) ->
  vector_sum(H, T).

vector_sum(First, [H|T]) ->
  New = lists:zipwith(fun(X,Y) -> X+Y end, First, H),
  vector_sum([New|T]).

matrix_map(F, Matrix) ->
  lists:map(fun(X) -> lists:map(F, X) end, Matrix).
% cov_data(W, N, K, Key)
%   Create random data for testing cov(W, Key).
%   The data will have a total of N vectors, distributed across the workers of W.
%   Each of these vectors will have K elements.  Each element is distributed
%   uniformly in [0,1].  This may not be the most useful distribution, but it
%   is a good start for some tests.
cov_data(W, N, K, Key) ->
  workers:broadcast(W,
    fun(ProcState, MyN) ->
      workers:put(ProcState, Key,
        [ misc:rlist(K, 1.0) || _ <- lists:seq(1,MyN) ])
    end,
    [ Hi-Lo || {Lo, Hi} <- misc:intervals(0, N, length(W)) ]).

% close(X, Y, Epsilon) -> Close
%   Test if X and Y are close.  If X and Y are numerical values, then they
%   are "close" if the difference between them is less Epsilon.  We try both
%   the relative and absolute difference, and accept if either is less than
%   Epsilon.  If X and Y are both lists or both tuples, then we recursively
%   check that all of their elements are close.
close(X, X, _) -> true;
close(X, Y, _) when is_integer(X), is_integer(Y) -> false;  % x /= y by previous pattern
close(X, Y, Epsilon) when is_number(X), is_number(Y) -> % one or both of X and Y are floats
  try
    abs(X-Y) =< Epsilon*max(max(abs(X), abs(Y)), 1)
  catch error:badarith -> % either X or Y is an integer that is too big to represent as a float
    false
  end;
close([HdX | TlX], [HdY | TlY], Epsilon) ->
  close(HdX, HdY, Epsilon) andalso close(TlX, TlY, Epsilon);
close(X, Y, Epsilon) when is_tuple(X), is_tuple(Y), tuple_size(X) == tuple_size(Y) ->
  close(tuple_to_list(X), tuple_to_list(Y), Epsilon);
close(_,_,_) -> false.

% close(X,Y) -> Close
%   Equivalent to close(X, Y, 1.0e-12).
close(X, Y) -> close(X, Y, 1.0e-12).


% cov_check(NProcs, Data) -> true, false, {Error, Reason}
%   Compute the covariance matrix for Data using the sequential version, cov(Data)
%   and the parallel version.  For the parallel version, create a tree of NProces
%   processes and use that.  Terminate the processes in this tree when done.
%
%   Return value:
%     true: if both processes compute a value, and the values from the
%       sequential and parallel versions are close.
%     false: if both processes compute a value, and the values from the
%       sequential and parallel versions are not close.
%     {Error, Reason}: one of the computations threw an error.  We
%       return the exception.
cov_check(NProcs, Data) when is_integer(NProcs) ->
  W = wtree:create(NProcs),
  Outcome = cov_check(W, Data),
  wtree:reap(W), % clean-up
  Outcome;

% cov_check(W, Data) -> true, false, {Error, Reason}
%   Like cov_check(NProcs, Data) but W is a worker tree (and not an integer)
%   so we don't need to create a worker tree, nor do we reap the tree when we
%   are done.  Note: we store the data for the test in the worker processes
%   with the key 'cov_check:data'.  If the workers already had a value
%   associated with that key, it will be overwritten.
cov_check(W, Data) ->
  Key = 'cov_check:data',
  try
    workers:update(W, Key, misc:cut(Data, W)),
    close(cov(W, Key), cov(Data))
  catch Error:Reason ->
    {Error, Reason}
  end.

% cov_check(W, N, K): like cov_check/2, but we create a random list of
%   N vectors with K elements per vector.  W can be a worker tree or
%   a positive integer.
cov_check(W, N, K) ->
  cov_check(W, [ misc:rlist(K, 1.0) || _ <- lists:seq(1,N) ]).


% your_answer(Who, Args): a place-holder for where you need to provide a solution
your_answer(Who, _Args) ->
  error({Who, missing_implementation}).
