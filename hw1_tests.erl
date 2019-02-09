-module(hw1_tests).
-import(hw1, [kth_largest/2, cov/1, cov/2, close/2, cov_check/2]).
-import(misc, [rlist/1, cut/2]).
-import(workers, [update/1]).
-import(wtree, [create/1]).
-include_lib("eunit/include/eunit.hrl").

kth_largest_test_() ->
  [?_assert(kth_largest(1, [1]) =:= 1),
  ?_assert(kth_largest(1, [1, 2, 3]) =:= 3),
  ?_assert(kth_largest(2, [1, 2, 3]) =:= 2),
  ?_assert(kth_largest(2, [1, 2, 2]) =:= 2),
  ?_assertException(error, function_clause, kth_largest(-1, [1])),
  ?_assertException(error, function_clause, kth_largest(2, [1])),
  ?_assertException(error, function_clause, kth_largest(1, foo)),
  ?_assertException(error, function_clause, kth_largest(foo, 1))].

% I should really have some larger tests for sequential cov
% but those are tedious to write expected values for
cov_test_() ->
  [?_assert(close(cov([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [[9, 9, 9], [9, 9, 9], [9, 9, 9]])),
  ?_assert(close(cov([[1, 2, 3], [3, 2, 1], [-4, -1, -6]]), [[13, 6, 15], [6, 3, 8], [15, 8, 67/3]])),
  ?_assert(close(cov([[1, 2], [4, 2], [2, 2], [7, 4], [5, 2]]), [[57/10, 8/5], [8/5, 4/5]])),
  ?_assertException(error, function_clause, cov([])),
  ?_assertException(error, function_clause, cov([[1, 2, 3]])), % div by 0 from N = 1 - 1
  ?_assertException(error, function_clause, cov([1, 2, 3])),
  ?_assertException(error, function_clause, cov([[1, 2, 3], [4, 6]])),
  ?_assertException(error, function_clause, cov([[1, 2, 3], foo]))]. 

cov_par_test_() ->
  Data_small = [misc:rlist(5) || _ <- lists:seq(1,100)],
  Data_large = [misc:rlist(25) || _ <- lists:seq(1,10000)],
  Data_tall  = [misc:rlist(50) || _ <- lists:seq(1,100)],
  Data_wide  = [misc:rlist(5) || _ <- lists:seq(1,10000)],
  W = wtree:create(2),
  workers:update(W, data, misc:cut(Data_small, W)), % Dummy worker/data
  [?_assert(cov_check(2, Data_small)),
  ?_assert(cov_check(2, Data_large)),
  ?_assert(cov_check(2, Data_tall)),
  ?_assert(cov_check(2, Data_wide)),
  ?_assert(cov_check(200, Data_small)),
  ?_assert(cov_check(200, Data_large)),
  ?_assert(cov_check(200, Data_tall)),
  ?_assert(cov_check(200, Data_wide)),
  ?_assertException(throw, {fail, _, _}, cov(W, foo)),
  ?_assertException(error, function_clause, cov(123, data))]. % div by 0 from N = 1 - 1