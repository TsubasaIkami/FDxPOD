function z = shrinkage(a,b)
z = max([(1-b/norm(a)) 0])*a;
end