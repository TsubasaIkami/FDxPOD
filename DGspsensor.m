% optimal sensor placement function

% input
% U: POD modes
% p: the number of the optimal sensors

% output
% sensor: index of the selected sensor positions

% REF: Y. Saito et al., “Determinant-based fast greedy sensor selection algorithm,” IEEE Access 9, 68535–68551 (2021).

function sensor = DGspsensor(U,p)

r = size(U,2);
n = size(U,1);
J = zeros(1,n);
sensor=zeros(1,p);

C = zeros(p,r);
i = 1;
parfor nn = 1 : n
    temp = U(nn,:);
    J(nn) = temp * temp';
end
[~,sensor(i)] = max(J); 
C(i,:) = U(sensor(i),:);
U(sensor(i),:) = 0;
CCTinv = inv(C(1:i,:)*C(1:i,:)');

if r >= p
    for i = 2 : p
        disp(i)
        Y = eye(r)-C(1:i-1,:)'*CCTinv(1:i-1,1:i-1)*C(1:i-1,:);
        parfor nn = 1 : n
            temp = U(nn,:);
            J(nn) = temp * Y * temp';
        end
        [~,sensor(i)] = max(J);
        C(i,:) = U(sensor(i),:);
        U(sensor(i),:) = 0;
        CCTinv = inv(C(1:i,:)*C(1:i,:)');
    end
end

if r < p
    for i = 2 : r
        disp(i)
        Y = eye(r)-C(1:i-1,:)'*CCTinv(1:i-1,1:i-1)*C(1:i-1,:);
        parfor nn = 1 : n
            temp = U(nn,:);
            J(nn) = temp * Y * temp';
        end
        [~,sensor(i)] = max(J);
        C(i,:) = U(sensor(i),:);
        U(sensor(i),:) = 0;
        CCTinv = inv(C(1:i,:)*C(1:i,:)');
    end
    clear CCTinv

    CTCinv = inv(C(1:i,:)'*C(1:i,:));
    for i = r+1 : p
        disp(i)
        parfor nn = 1 : n
            temp = U(nn,:);
            J(nn) = 1 + temp * CTCinv * temp';
        end
        [~,sensor(i)] = max(J);
        C(i,:) = U(sensor(i),:);
        u = U(sensor(i),:);
        CTCinv = CTCinv*(eye(r,r)- u' * inv(1+u*CTCinv*u') * u *CTCinv);
        U(sensor(i),:) = 0;
    end
end

end