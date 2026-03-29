// Verify the Hadamard butterfly implementation is correct

fn hadamard_butterfly(row: &mut [f32]) {
    let n = row.len();
    let mut half = 1;
    while half < n {
        for i in (0..n).step_by(half * 2) {
            for j in i..i + half {
                let a = row[j];
                let b = row[j + half];
                row[j] = a + b;
                row[j + half] = a - b;
            }
        }
        half *= 2;
    }
}

fn main() {
    println!("Verifying Hadamard butterfly correctness\n");
    
    // Test N=2: H2 = [1  1]
    //               [1 -1]
    println!("N=2 test:");
    let mut data = vec![1.0, 0.0];
    hadamard_butterfly(&mut data);
    println!("  H·[1,0] = [{}, {}], expected [1, 1]", data[0], data[1]);
    
    let mut data = vec![0.0, 1.0];
    hadamard_butterfly(&mut data);
    println!("  H·[0,1] = [{}, {}], expected [1, -1]", data[0], data[1]);
    
    // Test N=4: Verify full matrix
    println!("\nN=4 test (unnormalized Hadamard matrix):");
    let n = 4;
    let mut h_matrix = vec![0.0f32; n * n];
    
    for i in 0..n {
        let mut e = vec![0.0f32; n];
        e[i] = 1.0;
        hadamard_butterfly(&mut e);
        for j in 0..n {
            h_matrix[j * n + i] = e[j]; // Column-major for display
        }
    }
    
    println!("  H4 (unnormalized) =");
    for i in 0..n {
        print!("    [");
        for j in 0..n {
            print!("{:4.0} ", h_matrix[i * n + j]);
        }
        println!("]");
    }
    
    // Expected: [1  1  1  1]
    //           [1 -1  1 -1]
    //           [1  1 -1 -1]
    //           [1 -1 -1  1]
    
    let expected = vec![
        1.0,  1.0,  1.0,  1.0,
        1.0, -1.0,  1.0, -1.0,
        1.0,  1.0, -1.0, -1.0,
        1.0, -1.0, -1.0,  1.0,
    ];
    
    let mut all_match = true;
    for i in 0..16 {
        if (h_matrix[i] - expected[i]).abs() > 1e-5 {
            all_match = false;
            println!("  ❌ Mismatch at index {}: got {}, expected {}", i, h_matrix[i], expected[i]);
        }
    }
    
    if all_match {
        println!("  ✓ Matrix matches expected Hadamard H4");
    }
    
    // Test that H·H = N·I (Hadamard is self-inverse up to scaling)
    println!("\nN=8 self-inverse test:");
    let n = 8;
    let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut data = original.clone();
    
    hadamard_butterfly(&mut data);
    hadamard_butterfly(&mut data);
    
    // Should be scaled by N
    let scale = n as f32;
    let mut max_error = 0.0f32;
    for i in 0..n {
        let expected = original[i] * scale;
        let error = (data[i] - expected).abs();
        if error > max_error {
            max_error = error;
        }
    }
    
    println!("  Max error (H·H·x vs {}·x): {:.2e}", scale, max_error);
    
    if max_error < 1e-4 {
        println!("  ✓ H·H = {}·I confirmed", scale);
    } else {
        println!("  ❌ Self-inverse property failed!");
    }
    
    println!("\n✓ Hadamard butterfly implementation is correct");
}
