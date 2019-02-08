-module(hw1_tests).
-import(hw1, [kth_largest/2, cov/1, cov/2, close/2]).
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

cov_test_() ->
  [?_assert(close(cov([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [[9, 9, 9], [9, 9, 9], [9, 9, 9]])),
  ?_assert(close(cov([[1, 2, 3], [3, 2, 1], [-4, -1, -6]]), [[13, 6, 15], [6, 3, 8], [15, 8, 67/3]])),
  ?_assert(close(cov([[1, 2], [4, 2], [2, 2], [7, 4], [5, 2]]), [[57/10, 8/5], [8/5, 4/5]]))].