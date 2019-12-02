figure; 
addpath(genpath('../../../'))
for i = 0:9

load(sprintf('mnistD_DLE %d.mat', i))
[t, idx] = sort(ll);
for j = 1:1
    subplot(6,10,j+i*1)
    imshow(reshape(XData0(:,idx(end-j)),28,28)', 'InitialMagnification', 8000)
end
load(sprintf('mnistD_KSD %d.mat', i))
[t, idx] = sort(ll);
for j = 1:1
    subplot(6,10,j+10+i*1)
    imshow(reshape(XData0(:,idx(end-j)),28,28)', 'InitialMagnification', 8000)
end
load(sprintf('mnistD_NCE %d.mat', i))
[t, idx] = sort(ll);
for j = 1:1
    subplot(6,10,j+20+i*1)
    imshow(reshape(XData0(:,idx(end-j)),28,28)', 'InitialMagnification', 8000)
end

load(sprintf('mnistD_DLE %d.mat', i))
[t, idx] = sort(ll);
for j = 1:1
    subplot(6,10,j+30+i*1)
    c = reshape(XData0(:,idx(j)),28,28)';
    imshow(c, 'InitialMagnification', 8000)
end
load(sprintf('mnistD_KSD %d.mat', i))
[t, idx] = sort(ll);
for j = 1:1
    subplot(6,10,j+40+i*1)
    c = reshape(XData0(:,idx(j)),28,28)';
    imshow(c, 'InitialMagnification', 8000)
end
load(sprintf('mnistD_NCE %d.mat', i))
[t, idx] = sort(ll);
for j = 1:1
    subplot(6,10,j+50+i*1)
    c = reshape(XData0(:,idx(j)),28,28)';
    imshow(c, 'InitialMagnification', 8000)
end

end