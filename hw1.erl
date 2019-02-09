-module(hw1).

-export([kth_largest/2, worst_case/2]).  % for Q1
-export([count_make/0, count_make/1, count_proc/1, count_next/1, count_end/1]). % for Q2
-export([mean/1, esq/1, cov/1, cov/2]). % for Q3
-export([cov_table/0, pcov_table/0, kth_table/0]).
-export([close/2, close/3, cov_check/2, cov_check/3, cov_data/4, cov_timer/3]). % help with testing cov(W, Key)
-export([your_answer/2]). % throws an error if you try executing one of the stubs
-import(misc, [rlist/1, cut/2]).
-import(workers, [update/3]).

kth_largest(K, List) when is_integer(K) andalso K > 0 andalso is_list(List) andalso length(List) >= K-> 
  Acc = lists:sort(lists:sublist(List, K)),
  NewL = lists:nthtail(K, List),
  [KthElem|_] = kth2(NewL, Acc),
  KthElem.

kth2([H|T], Acc) -> % N times
  kth2(T, maintain(H, Acc));
kth2([], Acc) -> 
  Acc.

maintain(X, List = [H|_]) when X < H -> % Smaller than first elem
  List;
maintain(X, [_|T]) -> 
  insert(X, T).

insert(X, []) ->
  [X];

insert(X, [H|T]) when X =< H ->
  [X,H|T];

insert(X, [H|T]) ->
  [H|insert(X, T)].

worst_case(N, K) ->
  List = lists:seq(1, N), % Ascending list
  kth_largest(K, List).

% Functions for Q2

count_make(InitialCount) when is_integer(InitialCount) ->
  spawn(?MODULE, count_proc, [InitialCount]).

count_make() ->
  count_make(0).

count_proc(CurrentCount) ->
  receive
    {count, ReplyTo} ->
      ReplyTo ! {count, self(), CurrentCount},
      count_proc(CurrentCount + 1);
    exit ->
      ok
  end.

count_next(CounterPid) when is_pid(CounterPid) ->
  CounterPid ! {count, self()},
  receive
    {count, CounterPid, CurrentCount} ->
      CurrentCount
  end.

count_end(CounterPid) when is_pid(CounterPid) ->
  CounterPid ! exit.

% Functions for Q3

mean(List) ->
  N = length(List),
  [X/N || X <- vm_sum(List)].

% ESQ (and friends)

esq(X) when length(X) > 0 -> % Avoid div by 0
  esq(X, 0, divide).

esq(X, Offset, divide) when length(X) > Offset -> % Avoid div by 0
  N = length(X) + Offset,
  Matrices = [esq_matrix_single_col(Col) || Col <- X],
  Matrix = vm_sum(Matrices),
  matrix_map(fun(Y) -> Y/N end, Matrix);

esq(X, Offset, nodivide) when length(X) > Offset -> % Avoid div by 0
  Matrices = [esq_matrix_single_col(Col) || Col <- X],
  vm_sum(Matrices).

esq_matrix_single_col(Col) ->
  [[J1 * J2 || J1 <- Col] || J2 <- Col].

cov(X) when length(X) > 1 -> % Need > 1 to avoid div by 0 error
  N = length(X),
  Means = mean(X),
  Mean_map = mean_map(Means, N),
  Esq = esq(X, -1, divide),
  vm_sum([Esq, Mean_map]).

mean_map(Means, N) -> 
  [[ -((N/(N-1)) * (M1 * M2)) || M2 <- Means] || M1 <- Means].

% Parallel cov

cov(W, Key) ->
  wtree:reduce(W,
    fun(ProcState) -> cov_leaf(wtree:get(ProcState, Key)) end,  % Leaf
    fun(Left, Right) -> cov_combine(Left, Right) end, % Combine
    fun(Result) -> cov_root(Result) end
  ).

cov_leaf([]) -> none;

cov_leaf(Matrix) -> 
  N = length(Matrix),
  Var_sums = esq(Matrix, 0, nodivide),
  Mean_sums = vm_sum(Matrix),
  {Mean_sums, Var_sums, N}.

cov_root({Means, Vars, N}) -> 
  Final_means = lists:map(fun(X) -> X/N end, Means),
  Mean_map = mean_map(Final_means, N),
  Div_by = N - 1, % My syntax highlighting goes nuts if this is inside the anon function
  Esq = matrix_map(fun(X) -> X/Div_by end, Vars),
  vm_sum([Esq, Mean_map]).

cov_combine(Left, Right) ->
  case {Left, Right} of
    {Left, none} -> Left;
    {none, Right} -> Right;
    {Left, Right} -> combine(Left, Right)
  end.

combine({Mean1, Var1, N1}, {Mean2, Var2, N2}) ->
  Mean_sums = vm_sum([Mean1, Mean2]),
  Var_sums = lists:zipwith(fun(X,Y) -> vm_sum([X,Y]) end, Var1, Var2),
  {Mean_sums, Var_sums, N1 + N2}.

% Helpers

vm_sum([]) ->
  [];

vm_sum([H|[]]) ->
  H;

vm_sum([H|T]) ->
  vm_sum(H, T).

vm_sum(First = [H1|_], [H2|T2]) when is_list(H1) ->
  New = lists:zipwith(fun(X,Y) -> vm_sum([X, Y]) end, First, H2),
  vm_sum([New|T2]);

vm_sum(First, [H|T]) ->
  New = lists:zipwith(fun(X,Y) -> X+Y end, First, H),
  vm_sum([New|T]).

matrix_map(F, Matrix) ->
  lists:map(fun(X) -> lists:map(F, X) end, Matrix).








cov_table() ->
  N_consistent = 20000,
  Ks = lists:seq(10, 100, 10),
  K_data = [[misc:rlist(K) || _ <- lists:seq(1, N_consistent)] || K <- Ks],
  %K_consistent = 20,
  %Ns = lists:seq(50000, 500000, 50000),
  %N_data = [[misc:rlist(K_consistent) || _ <- lists:seq(1, N)] || N <- Ns],
  time_cov(K_data).


time_cov([]) -> ok;

time_cov([H|T]) ->
  Time = {time_it:t(fun() -> cov(H) end)},
  io:format("~p~n", [Time]),
  time_cov(T).

pcov_table() ->
  Data = [misc:rlist(30) || _ <- lists:seq(1, 50000)],
  Ps = [1, 2, 3, 4, 5, 6, 12, 24, 36, 48],
  time_pcov(Ps, Data),
  ok.

time_pcov([], _) -> ok;

time_pcov([H|T], Data) ->
  W = wtree:create(H),
  workers:update(W, data, misc:cut(Data, W)),
  Time = time_it:t(fun() -> cov(W, data) end),
  io:format("~p~n", [Time]),
  wtree:reap(W),
  time_pcov(T, Data).
  

kth_table() ->
  Ns = lists:seq(50000, 500000, 50000),
  Ks = lists:seq(100, 1000, 100),
  time_kth(Ks).

time_kth([]) -> ok;

time_kth([H|T]) ->
  Data = lists:seq(1, 250000),
  Time = time_it:t(fun() -> kth_largest(H, Data) end),
  io:format("~p~n", [Time]),
  time_kth(T).
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

cov_timer(N, K, P) ->
  W = wtree:create(P),
  Matrix = [misc:rlist(K) || _ <- lists:seq(1,N)],
  workers:update(W, data, misc:cut(Matrix, W)),
  wtree:retrieve(W, fun(_) -> ok end),
  time_it:t(fun() -> cov(W, data) end).