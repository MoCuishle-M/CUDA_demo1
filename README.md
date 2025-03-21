这是一个学习CUDA的demo。

使用cmake编译项目。
里面把CUDA版本写成死成了12.5，应该是可以自己改的。但如果没记错的话，有些版本不支持一些特性。
使用了O2优化。
最大cpp版本：理论上使用nvcc能支持的最大版本。
只写了5种CUDA计算能力，在CMakeLists.txt line14 ~15.其实这个是指CUDA的架构，也即Pascal、Volta、Turing、Ampere、Lovelace、Hopper等。CUDA的相关文档有些。
