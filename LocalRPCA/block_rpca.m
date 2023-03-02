% Copyright  (C) 2021, Aryaman Sinha
function [L,S] = blockRPCA(img,n,lambda,mu)
    [N,M] = size(img);
    if mod(n,2)~=0
        ws = floor(n/2);
        x = ceil(N/n)*n-N;
        y = ceil(M/n)*n-M;
        img_pad = zeros(N+x,M+y);
        L_pad = zeros(size(img));
        S_pad = zeros(size(img));
        img_pad(1:N,1:M) = img;
        for i=ws+1:n:N+x-ws
            for j=ws+1:n:M+y-ws
                block = img_pad(i-ws:i+ws,j-ws:j+ws);
                [l,s] = RobustPCA(block,lambda, mu);
                L_pad(i-ws:i+ws,j-ws:j+ws) = l;
                S_pad(i-ws:i+ws,j-ws:j+ws) = s;
            end
        end
        L = L_pad(1:N,1:M);
        S = S_pad(1:N,1:M);
    else
        ws = n;
        L = zeros(size(img));
        S = zeros(size(img));
        for i=1:N/ws
            for j=1:M/ws
                block = img(ws*i-ws+1:ws*i,ws*j-ws+1:ws*j);
                [l,s] = RobustPCA(block,lambda, mu);
                L(ws*i-ws+1:ws*i,ws*j-ws+1:ws*j) = l;
                S(ws*i-ws+1:ws*i,ws*j-ws+1:ws*j) = s;
            end
        end
    end
end
