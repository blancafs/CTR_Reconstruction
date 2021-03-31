
values = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
for i = values
    name = 'lfscipy3_rob%d.mat';
    name = sprintf(name, i);
    rob = load(name, '-ASCII');
    disp(rob);
    x = rob(:,1);
    y = rob(:,2);
    z = rob(:,3);
    pointsize = 8;  %adjust at will
    figure('name',  name');
    hold on 
        scatter3(x(:), y(:), z(:), pointsize);
    hold off
    title(i);
end