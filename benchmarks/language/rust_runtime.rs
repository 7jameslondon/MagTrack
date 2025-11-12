// build with: rustc ".\benchmarks\language\rust_runtime.rs" -O -C opt-level=3 -C lto=thin -C codegen-units=1 -C panic=abort -C target-cpu=native -C target-feature=+avx2,+fma -o ".\rust_runtime.exe"

mod center_of_mass;

use center_of_mass::center_of_mass;
use std::time::Instant;

fn main() {
    let width = 100usize;
    let height = 100usize;
    let n_images = 100usize;
    let image_size = width * height;

    let stack = vec![1.0_f64; image_size * n_images];
    let mut x = vec![0.0_f64; n_images];
    let mut y = vec![0.0_f64; n_images];

    for _ in 0..100 {
        center_of_mass(&stack, width, height, n_images, "none", &mut x, &mut y);
    }

    let start = Instant::now();
    for _ in 0..10000 {
        center_of_mass(&stack, width, height, n_images, "none", &mut x, &mut y);
    }
    let elapsed = start.elapsed();

    println!("Runtime: {:.6} seconds", elapsed.as_secs_f64());
}
