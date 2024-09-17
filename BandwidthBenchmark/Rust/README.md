# BandwidthBenchmark in Rust

Implementation of [TheBandwidthBenchmark](https://github.com/RRZE-HPC/TheBandwidthBenchmark) in Rust with multi-threading support.

This is a collection of simple streaming kernels.

Apart from the micro-benchmark functionality this is also a blueprint for other micro-benchmark applications.

Output is similar to C-version for compatibility.

# Target-triple and Target-cpu

You can set your own target-triple and target-cpu in **./cargo/config.toml**.
The reason for specifying the target-triple and target-cpu is to be able to generate optimal assembly with all the instructions supported by your cpu architecture.

Usually a list of target-features for a specific target-triple and target-cpu can be listed using following command:

```
rustc --print cfg -C target-cpu=native -C opt-level=3
```

By default, the target-triple is:
```
[target.x86-64-unknown-linux-gnu]
rustflags = [
    "-C",
    "target-cpu=native",
    "-C",
    "opt-level=3",
]
```

# Building and running the program
It is fairly simple to run the program.

A binary named **bench** can be built using :
```
cargo b --release
```
This command will output **bench** binary in ./target/release.
Then you can juse use 
```
cargo r --release
```
The second option to build a binary is to use Makefile commands.
```
make
```
comand will output **bench** binary in the ./ directory i.e. the current folder.
Then you can juse use 
```
./bench
```

The binary takes 3 parameters : **-n, -size, -ntimes** which are explained below:

```
Usage: ./bench [OPTIONS]
or
Usgae: cargo r --release -- [OPTIONS]

Options:
  -n, --n <N>
          Number of threads
          
          [default: max #threads available on your machine]

  -s, --size <SIZE>
          Size of the total dataset in bytes
          
          [default: 120000000]

  -n, --ntimes <NTIMES>
          Number of time to run all the benchmarks
          
          [default: 10]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

If you just use 
```
./bench
```
the program will run in multi-threaded fashion with max available cores on you CPU and with 120000000 bytes of data per vector.

If you wish to run the program serially, run the below command:
```
./bench -n 1
```

# Assembly Output
You can generate assembly either for the whole code or just for specific kernel.

1. To generate assembly for the whole program, use below command:
```
make asm 
```
2. To generate assembly specific to a kernel, please make sure that [cargo-show-asm](https://crates.io/crates/cargo-show-asm) is installed. Then use the following command:
```
cargo asm bench::copy --rust
```

**Note :** To make assembly available for a specific kernels, **#[inline(never)]** is specificed above the kernel. 

# Output
A sample output from the benchmark is shown below:

```
Benchmarking with 8 threads.
Total allocated datasize: 3840.00 MB.
Initialization of arrays took : 506.008814ms.
----------------------------------------------------------------------------------------------------------
Function        | Rate(MB/s)      | Rate(MFlop/s)   | Avg time       | Min time        | Max time        |
----------------------------------------------------------------------------------------------------------
Init:           | 8923.15         | -               | 0.1120         | 0.1076          | 0.1372          |
Sum:            | 19562.93        | 2445.37         | 0.0549         | 0.0491          | 0.0883          |
Copy:           | 11859.23        | -               | 0.1655         | 0.1619          | 0.1868          |
Update:         | 17723.62        | 1107.73         | 0.1100         | 0.1083          | 0.1143          |
Triad:          | 13162.47        | 1096.87         | 0.2207         | 0.2188          | 0.2255          |
Daxpy:          | 18254.28        | 1521.19         | 0.1604         | 0.1578          | 0.1643          |
STriad:         | 14149.89        | 884.37          | 0.2732         | 0.2714          | 0.2819          |
SDaxpy:         | 17545.77        | 1096.61         | 0.2219         | 0.2189          | 0.2240          |
----------------------------------------------------------------------------------------------------------
Solution Validates
```
