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
6.  **Code Complexity and Factory Pattern**: Implemented a complex MWE with `AbstractNode` template hierarchy, `DagIndexer` stub, `ConstMinusVar` logic, and 20+ dummy node types using a factory pattern to mimic the library's AST construction. Added dummy functions to increase code size. Result: Correct output.
7.  **`__attribute__((flatten))`**: Attempted to force inlining using `__attribute__((flatten))` on `ConstMinusVarNode::evalC0` to mimic potential aggressive inlining by `-O2`. Result: Correct output.
8.  **Implicit Type Conversion**: Strictly matched `int` vs `unsigned` usage for `coeffNo` between `AbstractNode` virtual interface and `ConstMinusVar` implementation, to trigger any potential implicit conversion bugs. Result: Correct output.
9.  **`eval` and `evalHelper` Mimicry**: Added dummy `eval` and `evalHelper` functions to `ConstMinusVar` namespace to match the `EvalSub.h` structure, even if unused in the main path. Result: Correct output.
10. **Large Graph Execution**: Executed the solver loop on a graph with 50+ nodes including dummy operations to simulate a more complex runtime environment. Result: Correct output.

Despite these efforts, the single-file MWE consistently produces the correct result with `g++-10 -O2`. This suggests that the bug depends on factors not easily replicated in a clean MWE, such as:
*   **Code Complexity / Inlining Heuristics**: The `DMapProblem` application involves parsing, graph construction, and a complex class hierarchy. The size of the translation unit and the complexity of the surrounding code likely trigger a specific inlining decision in `g++-10`.
*   **Compilation Unit Boundaries**: The interaction between templates instantiated in the main file and other translation units might be relevant.

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
1.  **Analyze Assembly**: The most reliable way forward is to compare the assembly of `DMapProblem.o` (buggy) and `mwe.o` (correct) to pinpoint exactly how `ConstMinusVar::evalC0` is being inlined and where the optimization diverges.
2.  **Use Library Headers (Stripping Down)**: Instead of mocking, create a MWE that includes the *actual* library headers but only uses a minimal subset of functionality, gradually stripping away includes until the bug disappears. This "stripping down" approach might be more effective than "building up".
3.  **Compiler Internal State**: The bug might depend on the internal state of the compiler (e.g., memory usage, symbol table order) which is hard to replicate with clean code.
