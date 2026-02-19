# Bug Report: Incorrect Results with GCC 10 and -O2

## Description
The program `DMapProblem` produces incorrect results when compiled with `g++-10` and `-O2` optimization.
The issue is localized to `include/capd/autodiff/EvalSub.h` in `ConstMinusVar::evalC0`.
The incorrect output appears for higher order coefficients (starting from index 3 in the example).

## Reproduction Steps
1.  Compile `DMapProblem` with `g++-10 -O2`:
    ```bash
    make clean && make CXX=g++-10
    ./DMapProblem
    ```
2.  Observe incorrect output:
    ```
    {2}
    {1}
    {-0.5}
    {0}
    {0}
    {0}
    ```
    Expected output:
    ```
    {2}
    {1}
    {-0.5}
    {0.166667}
    {-0.0416667}
    {0.00833333}
    ```

## Attempts to Isolate (MWE)
I attempted to create a Minimal Working Example (MWE) in a single `.cpp` file to reproduce the bug.
Strategies tried:
1.  **Simple Scalar MWE**: Implemented `ConstMinusVar::evalC0` with `double*` pointers and a simple loop mimicking the ODE solver. Result: Correct output.
2.  **Vector Class Simulation**: Implemented a `Vector` struct to mimic `capd::vectalg::Vector` with operator overloading and dynamic memory allocation. Result: Correct output.
3.  **Flat Memory Layout**: Simulated the `DagIndexer` memory layout using a single large `std::vector<double>` and pointer arithmetic, matching the library's internal storage. Result: Correct output.
4.  **Type Exactness**: Matched `coeffNo` types (`int` vs `unsigned`) exactly as in the library headers. Result: Correct output.
5.  **Loop Structure**: Mimicked the `eval` loop iterating over a vector of `AbstractNode*`. Result: Correct output.

Despite these efforts, the single-file MWE consistently produces the correct result with `g++-10 -O2`. This suggests that the bug depends on:
*   **Code Complexity / Inlining Heuristics**: The `DMapProblem` application involves parsing, graph construction, and a complex class hierarchy (`Map`, `BasicFunction`, `DagIndexer`). The size of the translation unit and the complexity of the surrounding code likely trigger a specific inlining decision in `g++-10` that is not triggered in a small MWE.
*   **Compilation Unit Boundaries**: The library code is split across many headers and instantiated in `DMapProblem.cpp`. While the MWE is also a single TU, the lack of external dependencies and smaller code size might prevent the problematic optimization path.

## Fix Confirmation
The bug is confirmed to be related to **partial inlining**.
Compiling `DMapProblem` with `-fno-partial-inlining` fixes the issue:
```bash
make CXX="g++-10 -fno-partial-inlining"
./DMapProblem
```
Output:
```
{2}
{1}
{-0.5}
{0.166667}
{-0.0416667}
{0.00833333}
```

The suggested fix in `README.md` (using ternary operator) also avoids the problematic control flow that triggers the partial inlining bug.

## Future Search Propositions
To reproduce this in a MWE, one might need to:
1.  **Increase Code Complexity**: Add dummy functions or classes to the MWE to increase the code size and complexity, potentially influencing the inliner's cost model.
2.  **Force Specific Inlining**: Use `__attribute__((flatten))` or specific inlining attributes to mimic the library's inlining decisions.
3.  **Check Assembly**: Analyze the assembly output of `DMapProblem.o` (buggy) vs `mwe.o` (correct) to see exactly how `evalC0` is inlined and where the divergence occurs.
4.  **Split Translation Units**: Create a MWE with multiple `.cpp` files to mimic the library structure more closely (though the task required a single `.cpp` file).
