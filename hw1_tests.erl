-module(hw1_tests).
-import(hw1, [kth_largest/2]).
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