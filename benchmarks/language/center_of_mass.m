function [x, y] = center_of_mass(stack, background)
%CENTER_OF_MASS Calculate x and y centers of mass for a stack of images.
%   [X, Y] = CENTER_OF_MASS(STACK) computes the center of mass for each
%   2D image along the third dimension of STACK. STACK must be a 3D array
%   of size (N_PIXELS, N_PIXELS, N_IMAGES). The default background option
%   is 'none'.
%
%   [...] = CENTER_OF_MASS(STACK, BACKGROUND) allows selecting the
%   background subtraction strategy: 'none', 'mean', or 'median'.
%
%   This function mirrors the behaviour of magtrack.core.center_of_mass.
%
%   Parameters
%   ----------
%   stack : 3D float array, shape (n_pixels, n_pixels, n_images)
%       The image-stack. The images must be square.
%   background : string, optional
%       Background pre-processing. 'none' (default) uses the raw data.
%       'mean' subtracts the per-image mean. 'median' subtracts the
%       per-image median.
%
%   Returns
%   -------
%   x : 1D float array, shape (n_images, 1)
%       The x coordinates of the center.
%   y : 1D float array, shape (n_images, 1)
%       The y coordinates of the center.

    if nargin < 2 || isempty(background)
        background = 'none';
    end

    validatestring(background, {'none', 'mean', 'median'}, mfilename, 'background');

    if ~isfloat(stack)
        stack = double(stack);
    end

    stack_size = size(stack);
    width = stack_size(1);

    switch background
        case {'none'}
            stack_norm = stack;
        case {'mean'}
            frame_means = mean(mean(stack, 1), 2);
            stack_norm = stack - frame_means;
            stack_norm = abs(stack_norm);
        case {'median'}
            frame_medians = median(median(stack, 1), 2);
            stack_norm = stack - frame_medians;
            stack_norm = abs(stack_norm);
    end

    total_mass = squeeze(sum(sum(stack_norm, 1), 2));
    total_mass = reshape(total_mass, 1, []);
    total_mass(total_mass == 0) = NaN;

    index = (0:width-1)';

    sum_over_rows = squeeze(sum(stack_norm, 1));
    if isvector(sum_over_rows)
        sum_over_rows = reshape(sum_over_rows, width, []);
    end
    x = sum(index .* sum_over_rows, 1) ./ total_mass;
    x = reshape(x, [], 1);

    sum_over_columns = squeeze(sum(stack_norm, 2));
    if isvector(sum_over_columns)
        sum_over_columns = reshape(sum_over_columns, width, []);
    end
    y = sum(index .* sum_over_columns, 1) ./ total_mass;
    y = reshape(y, [], 1);
end
