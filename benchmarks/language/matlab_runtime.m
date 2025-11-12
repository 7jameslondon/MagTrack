stack = ones(100, 100, 100);

% Warm up
for i = 1:100
    center_of_mass(stack);
end

% Runtime measurement
start_time = tic;
for i = 1:10000
    center_of_mass(stack);
end
elapsed_time = toc(start_time);
fprintf('Runtime: %.6f seconds\n', elapsed_time);
