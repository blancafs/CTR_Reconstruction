
rob = load('rob5.mat', '-ASCII');
disp(rob);
x = rob(:,1);
y = rob(:,2);
z = rob(:,3);
pointsize = 8;  %adjust at will
scatter3(x(:), y(:), z(:), pointsize);