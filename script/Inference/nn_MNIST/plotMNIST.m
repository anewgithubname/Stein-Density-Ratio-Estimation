figure; 
addpath(genpath('../../../'))
for i = 0:9

load(sprintf('mnistD_DLE %d.mat', i))
[t, idx] = sort(ll);
for j = 1:2
    subplot(6,20,j+i*2)
    imshow(reshape(XData0(:,idx(end-j)),28,28)', 'InitialMagnification', 8000)
end
load(sprintf('mnistD_KSD %d.mat', i))
[t, idx] = sort(ll);
for j = 1:2
    subplot(6,20,j+20+i*2)
    imshow(reshape(XData0(:,idx(end-j)),28,28)', 'InitialMagnification', 8000)
end
load(sprintf('mnistD_NCE %d.mat', i))
[t, idx] = sort(ll);
for j = 1:2
    subplot(6,20,j+40+i*2)
    imshow(reshape(XData0(:,idx(end-j)),28,28)', 'InitialMagnification', 8000)
end

load(sprintf('mnistD_DLE %d.mat', i))
[t, idx] = sort(ll);
for j = 1:2
    subplot(6,20,j+60+i*2)
    c = reshape(XData0(:,idx(j)),28,28)';
    imshow(c, 'InitialMagnification', 8000)
end
load(sprintf('mnistD_KSD %d.mat', i))
[t, idx] = sort(ll);
for j = 1:2
    subplot(6,20,j+80+i*2)
    c = reshape(XData0(:,idx(j)),28,28)';
    imshow(c, 'InitialMagnification', 8000)
end
load(sprintf('mnistD_NCE %d.mat', i))
[t, idx] = sort(ll);
for j = 1:2
    subplot(6,20,j+100+i*2)
    c = reshape(XData0(:,idx(j)),28,28)';
    imshow(c, 'InitialMagnification', 8000)
end

end