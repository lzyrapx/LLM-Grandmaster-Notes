#### 总结

在 K 维度较小的场景下，DeepGEMM 的性能优于 CUTLASS。主要原因：
- DeepGEMM 会使用算术强度更大的 QGMMA 指令。例如，CUTLASS 64x128x32 vs. DeepGEMM 64x160x32。
- CUTLASS 的 Epilogue 中进行了不必要的 LinearCombination，导致执行了大量的不必要的 FFMA 指令。