use std::cmp::Ordering;

/// Compute the center of mass for each image in a stack.
///
/// # Parameters
/// * `stack` - Flattened stack of images stored in row-major order.
/// * `width` - Width of each image.
/// * `height` - Height of each image.
/// * `n_images` - Number of images in the stack.
/// * `background` - One of "none", "mean", or "median" to control background correction.
/// * `x_out` - Output slice for x coordinates (length must be at least `n_images`).
/// * `y_out` - Output slice for y coordinates (length must be at least `n_images`).
pub fn center_of_mass(
    stack: &[f64],
    width: usize,
    height: usize,
    n_images: usize,
    background: &str,
    x_out: &mut [f64],
    y_out: &mut [f64],
) {
    if width == 0 || height == 0 || n_images == 0 {
        return;
    }

    let image_size = width * height;
    let required_len = image_size * n_images;
    debug_assert!(stack.len() >= required_len, "stack length is too small");
    debug_assert!(x_out.len() >= n_images, "x_out length is too small");
    debug_assert!(y_out.len() >= n_images, "y_out length is too small");

    let background_mode = match background {
        "none" => Background::None,
        "mean" => Background::Mean,
        "median" => Background::Median,
        _ => {
            for i in 0..n_images {
                x_out[i] = f64::NAN;
                y_out[i] = f64::NAN;
            }
            return;
        }
    };

    let mut column_sums = vec![0.0_f64; width];
    let mut median_buffer = if matches!(background_mode, Background::Median) {
        Some(vec![0.0_f64; image_size])
    } else {
        None
    };

    for image_idx in 0..n_images {
        let image_start = image_idx * image_size;
        let image = &stack[image_start..image_start + image_size];

        let mut total_mass = 0.0_f64;
        let mut x_numerator = 0.0_f64;
        let mut y_numerator = 0.0_f64;

        column_sums.fill(0.0);

        let background_value = match background_mode {
            Background::None => 0.0,
            Background::Mean => image.iter().copied().sum::<f64>() / image_size as f64,
            Background::Median => {
                let buffer = median_buffer.as_mut().expect("median buffer missing");
                buffer.copy_from_slice(image);
                buffer.sort_by(|a, b| match a.partial_cmp(b) {
                    Some(ordering) => ordering,
                    None => Ordering::Equal,
                });
                if image_size % 2 == 0 {
                    let mid = image_size / 2;
                    0.5 * (buffer[mid - 1] + buffer[mid])
                } else {
                    buffer[image_size / 2]
                }
            }
        };

        let subtract_background = matches!(background_mode, Background::Mean | Background::Median);
        for (row_idx, row) in image.chunks(width).enumerate() {
            let mut row_sum = 0.0_f64;

            if subtract_background {
                for (col_idx, &raw_value) in row.iter().enumerate() {
                    let value = (raw_value - background_value).abs();
                    row_sum += value;
                    column_sums[col_idx] += value;
                    total_mass += value;
                }
            } else {
                for (col_idx, &value) in row.iter().enumerate() {
                    row_sum += value;
                    column_sums[col_idx] += value;
                    total_mass += value;
                }
            }

            y_numerator += row_idx as f64 * row_sum;
        }

        for (col, &value) in column_sums.iter().enumerate() {
            x_numerator += col as f64 * value;
        }

        if total_mass == 0.0 {
            x_out[image_idx] = f64::NAN;
            y_out[image_idx] = f64::NAN;
        } else {
            x_out[image_idx] = x_numerator / total_mass;
            y_out[image_idx] = y_numerator / total_mass;
        }
    }
}

enum Background {
    None,
    Mean,
    Median,
}
