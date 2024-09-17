pub enum Benchmark {
    Init = 0,
    Sum,
    Copy,
    Update,
    Triad,
    Daxpy,
    Striad,
    Sdaxpy,
    Numbench,
}

pub struct BenchmarkType {
    pub label: String,
    pub words: usize,
    pub flops: usize,
}