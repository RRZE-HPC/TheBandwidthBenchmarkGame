use clap::Parser;
use std::fmt::Debug;
use std::thread::available_parallelism;

#[derive(Parser, Debug)]
#[command(
    version,
    about,
    long_about = "TheBandwidthBenchmarks written in Rust programming language"
)]
pub struct ArgParser {
    #[arg(
        short,
        long,
        default_value_t  = available_parallelism().unwrap().get() as usize,
        help = "Number of threads"
    )]
    pub n: usize,

    #[arg(
        short,
        long,
        default_value_t = 120000000usize,
        help = "Size of the total dataset in bytes"
    )]
    pub size: usize,

    #[arg(
        short,
        long,
        default_value_t = 10usize,
        help = "Number of time to run all the benchmarks"
    )]
    pub ntimes: usize,
}
