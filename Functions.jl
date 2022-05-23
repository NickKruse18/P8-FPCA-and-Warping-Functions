using LinearAlgebra
using Pkg
using Plots
using CSV
using DataFrames
using LightXML
using Distributions
using JLD

function InnerL2(x,y,t;typ="lin")
    if typ == "lin"
        x2 = size(x,2);  m = size(t,1);  A2 = t[2:m]-t[1:m-1];  A1 = 2*([A2; 0] + [0; A2])
        return (x'.*A1'+ [zeros(x2) x[1:m-1,:]'.*A2'] + [x[2:m,:]'.*A2' zeros(x2)])*y/6
    elseif typ == "con" m = size(t,1);  A2 = [t[2:m]-t[1:m-1]; 0];  return (x'.*A2')*y end
end

function GeneratePCA(Cov,t,n;mu=0,var=0,B=randn(size(Cov,2),n),E=randn(size(Cov,1),n)*var)
    X = Cov*B;  return mu .+ X + E
end

function GetPCA(X,t;d=1,pve=0,typ="lin",with_mu=true)
    x2 = size(X,2);  mu = X*ones(x2)/x2;  if with_mu X = X - mu*ones(1,x2) end
    if typ == "lin"
        m = size(t,1);  A2 = t[2:m]-t[1:m-1];  A1 = 2*([A2; 0] + [0; A2])
        C = X*(X'.*A1'+ [zeros(x2) X[1:m-1,:]'.*A2'] + [X[2:m,:]'.*A2' zeros(x2)])/6/x2
    elseif typ == "con" m = size(t,1);  A2 = [t[2:m]-t[1:m-1]; 0]
        C = X*(X'.*A2')/6/x2 end
    var = tr(C);  E = eigen(C);  ve = 0;  if (pve != 0) d = 0;  if (var != 0)
        for i in 1:m ve += real.(eigvals(E)[m-i+1]);  if ve/var > pve d = i;  break end end end end
    V = Real.(eigvecs(E)[:,m:-1:m-d+1]);  V = V./sqrt.(diag(InnerL2(V,V,t,typ=typ))')
    return [V,with_mu*mu,eigvals(E)]
end

function FitPCA(X,t;d=1,pve=0,typ="lin",with_mu=true,coefs=false)
    D = GetPCA(X,t,d=d,pve=pve,typ=typ,with_mu=with_mu); E = D[1]; mu = D[2]
    if coefs return [InnerL2(E,X-mu*ones(1,size(X,2)),t,typ=typ), E, mu, D[3]] else
        return mu*ones(1,size(X,2)) + E*InnerL2(E,X-mu*ones(1,size(X,2)),t,typ=typ) end
end

function ConfidenceInterval(Data,k1,k2,conf)
    n = trunc.(Int,size(Data,2)/k2)
    for i1 in 1:k1 for i2 in 1:k2 Data[i1,i2:k2:n*k2] = sort(Data[i1,i2:k2:n*k2]) end end
    if size(conf,1) == 2
        L = Data[:,(1:k2) .+ (trunc(Int,n*conf[1])-1)*k2];  U = Data[:,(1:k2) .+ trunc(Int,n*conf[2])*k2]
        return [trunc.(Int,0.5 .- (sign.(L) .* sign.(U))/2), L, U]
    else return Data[:,(1:k2) .+ trunc(Int,n*conf)*k2] end
end

function BootStrapSample(Fit,Res)
    Res = Res[:,((ones(1,size(Res,1))*Res).!=0)[:]];  R = trunc.(Int,ceil.(size(Res,2)*rand(size(Fit,2))));  return Fit + Res[:,R]
end

function BootStrapPCA(Fit,Res,E,t,n;typ="lin",with_mu=true)
    e2 = size(E,2);  Boot_Eig = zeros(e2,e2*n)
    for i in 1:n
        X_b = BootStrapSample(Fit,Res); D = FitPCA(X_b,t,d=e2,typ=typ,with_mu=with_mu,coefs=true)
        Err = D[2].*sign.(diag(D[2]'*E)')-E;  Boot_Eig[:,e2*(i-1)+1:e2*i] = Err'*Err/size(E,1) end
    return Boot_Eig
end

function BootStrapPCAWARVAR(Fit_P,Res_P,E_P,Fit_W,Res_W,E_W,t,n_P,n_V;B=zeros(0,size(Fit_P,2)),typ="lin",with_mu=true,Ind=(1:size(Fit_P,2)),season=false)
    ep2 = size(E_P,2);  ew2 = (E_W!=0)*size(E_W,2);  Boot_VAR = zeros(ep2+ew2+size(B,1),(ep2+ew2+size(B,1))*n_P*n_V)
    for i in 1:n_P
        X_P = BootStrapSample(Fit_P,Res_P);  D_P = FitPCA(X_P,t,d=ep2,typ=typ,with_mu=with_mu,coefs=true)
        if E_W != 0 X_W = BootStrapSample(Fit_W,Res_W);  D_W = FitPCA(X_W,t,d=ew2,with_mu=true,coefs=true)
            X_V = [sign.(diag(E_P'*D_P[2]))'.*D_P[1]; sign.(diag(E_W'*D_W[2]))'.*D_W[1]]
        else X_V = sign.(diag(E_P'*D_P[2])).*D_P[1] end;  if season X_V = CorrectSeasonality(X_V,Ind,3,true)[1] end;  V = FitVAR([X_V; B],Ind=Ind)
        if n_V == 1 Boot_VAR[:,(ep2+ew2+size(B,1))*n_V*(i-1)+1:(ep2+ew2+size(B,1))*n_V*i] = V[1] else
        Boot_VAR[:,(ep2+ew2+size(B,1))*n_V*(i-1)+1:(ep2+ew2+size(B,1))*n_V*i] = BootStrapVAR(V[2],[X_V; B]-V[2],V[1],n_V,Ind=Ind) end end
    return Boot_VAR
end

function ModelPCA(x,t,var,n,k)
    e2 = size(x,2);  E = GetPCA(x,t,d=e2,with_mu=false)[1];  Var_Eig = zeros(e2,e2*n)
    for i in 1:n X = GeneratePCA(x,t,k,var=var); D = FitPCA(X,t,d=e2,coefs=true)
        Err = D[2].*sign.(diag(D[2]'*E)')-E; Var_Eig[:,e2*(i-1)+1:e2*i] = Err'*Err/size(x,1) end
    return Var_Eig
end

function GenerateVAR(A,n;mu=0,var=1,B=var*randn(size(A,1),n),r=B.+mu,Ind=1:n)
    a = size(A,1);  for i in 2:n if ((Ind[i]-Ind[i-1]).==1) r[:,i] += A*r[:,i-1] end end;  return r
end

function FitVAR(X;with_mu=false,Ind=(1:size(X,2)),R=zeros(size(X,1),size(X,1)+with_mu))
    x1 = size(X,1);  x2 = size(X,2);  B = X[:,1:x2-1].*((Ind[2:x2]-Ind[1:x2-1]).==1)'
    S = [(Ind[2]-Ind[1] == 1)*X[:,1:1] X[:,2:x2-1].*(((Ind[3:x2]-Ind[2:x2-1]).==1).&((Ind[2:x2-1]-Ind[1:x2-2]).!=1))' zeros(x1)]
    X = X[:,2:x2].*(((Ind[2:x2]-Ind[1:x2-1]).==1)');  if with_mu B = [ones(1,size(X,2)-1); B] end
    if sum(R) > 0;  A = zeros(size(R));  for i in 1:x1 B_i = B[(R[i,:].==0),:]
        A[i:i,(R[i,:].==0)] = X[i:i,:]*B_i'*inv(B_i*B_i') end;  else A = X*B'*inv(B*B') end
    return [A,[zeros(x1) A*B]+S]
end

function BootStrapVAR(Fit,Res,A,n;with_mu=false,Ind=(1:size(Res,2)),R=zeros(size(A)))
    a = size(A,2);  Boot_A = zeros(a-with_mu,a*n);  x2 = size(Fit,2); X_b = 0
    for i in 1:n 
        X_b = [Fit[:,1] Fit[:,2:x2-1].*(((Ind[3:x2]-Ind[2:x2-1]).==1).&((Ind[2:x2-1]-Ind[1:x2-2]).!=1))' zeros(a-with_mu)]
        X_b[(ones(a)*[0; (Ind[2:x2]-Ind[1:x2-1])]'.==1)] = BootStrapSample(zeros(size(Res)),Res)[(ones(a)*[0; (Ind[2:x2]-Ind[1:x2-1])]'.==1)]
        X_b = GenerateVAR(A[:,1:a-with_mu],x2,B=X_b,Ind=Ind);  D = FitVAR(X_b,with_mu=with_mu,Ind=Ind,R=R)
        Boot_A[:,a*(i-1)+1:a*i] = D[1] end;  return Boot_A
end

function ApplyWarp(X,t,h,t_i;typ="lin")
    i = 2;  ti = 2;  remove = 2;  Xs = zeros(size(X,1)+size(h,1)-2);  ts = zeros(size(t,1)+size(t_i,1)-2)
    while (i < size(t,1)+1) & (ti < size(t_i,1)+1)
        if t[i] < h[ti] ts[i+ti-remove] = t_i[ti-1]+(t[i]-h[ti-1])*(t_i[ti]-t_i[ti-1])/(h[ti]-h[ti-1])
            Xs[i+ti-remove] = X[i];  i += 1
        elseif t[i] > h[ti] Xs[i+ti-remove] = X[i-1]+(typ=="lin")*(h[ti]-t[i-1])*(X[i]-X[i-1])/(t[i]-t[i-1])
            ts[i+ti-remove] = t_i[ti];  ti += 1
        else Xs[i+ti-remove] = X[i];  ts[i+ti-remove] = t_i[ti];  i += 1;  ti += 1;  remove += 1 end end
    return [Xs[1:i+ti-remove-1],ts[1:i+ti-remove-1]]
end

function FirstOrderWarp(X,Y,t,Phi,t_i;typ="lin")
    x2 = size(Phi,2);  m = size(t,1);  dX = (X[2:m]-X[1:m-1].+1)./(t[2:m]-t[1:m-1])
    A2 = (t[2:m]-t[1:m-1]).*(dX.^2);  A1 = 2*([A2; 0] + [0; A2])
    Chi = (Phi'.*A1'+ [zeros(x2) Phi[1:m-1,:]'.*A2'] + [Phi[2:m,:]'.*A2' zeros(x2)])*Phi/6
    if typ == "lin" A2 = (t[2:m]-t[1:m-1]).*dX;  A1 = 2*([A2; 0] + [0; A2])
        Gamma = (Phi'.*A1'+ [zeros(x2) Phi[1:m-1,:]'.*A2'] + [Phi[2:m,:]'.*A2' zeros(x2)])*(Y-X)/6
    elseif typ == "con" A2 = (t[2:m]-t[1:m-1]).*dX.*((Y-X)[1:m-1]);  A1 = [A2; 0] + [0; A2];  Gamma = Phi'*A1/2 end
    beta = inv(Chi)*Gamma;  beta0 = beta[:]; smooth = 0;  scale = 0.00001*tr(Chi);  ite = 1;  w = true
    while w w = false;  for i in 1:size(beta0,1) lower = (i>1)*beta0[i-(i>1)]-(t_i[i+1]-t_i[i]);  upper = 1-t_i[i+1]
            if (beta0[i]+0.0001 < lower) | (beta0[i]-0.0001 > upper) error = beta0[i]+t_i[i+1] - upper*(beta0[i]-0.0001 > upper) - lower*(beta0[i]+0.0001 < lower)
                smooth += scale*ite*abs(error);  w=true;  break end end
        beta0 = inv(Chi+smooth*I)*Gamma;  ite += ite end;
    return beta0
end

function ExtendPoints(X,t,t_i;typ="lin")
    t1 = size(t,1);  ti1 = size(t_i,1);  i1 = 1;  i2 = 1; remove = 0;  Xs = zeros(t1+ti1);  ts = zeros(t1+ti1);  i = 1
    while i < (size(Xs,1)-remove+1) I1 = i1-(i1>t1);  I2 = i2-(i2>ti1)
        if (i2<=ti1)*t[I1] < (i1<=t1)*t_i[I2] Xs[i] = X[i1];  ts[i] = t[i1];  i1 += 1 elseif (i2<=ti1)*t[I1] > (i1<=t1)*t_i[I2]
        I1 = I1+(i1==1);  Xs[i] = X[I1-1]+(typ=="lin")*(t_i[i2]-t[I1-1])*(X[I1]-X[I1-1])/(t[I1]-t[I1-1]);  ts[i] = t_i[i2];  i2 += 1 else
        Xs[i] = X[i1];  ts[i] = t[i1];  i1 += 1;  i2 += 1; remove += 1 end; i += 1 end
    return [Xs[1:size(Xs,1)-remove],ts[1:size(Xs,1)-remove]]
end

function LinearBasis(X,t,t_i;ex_ends=false)
    ti1 = size(t_i,1);  D = ExtendPoints(X,t,t_i);  X = D[1];  t = D[2];  Phi = zeros(size(t,1),ti1)
    for i in 1:ti1
        Phi[:,i] = ExtendPoints([zeros(min(i-1,2)); 1; zeros(min(ti1-i,2))],t_i[i-min(i-1,2):i+min(ti1-i,2)],t)[1]
    end
    if ex_ends return [Phi[:,2:ti1-1],t,X] else return [Phi,t,X] end
end

function TruncateFunction(X,t,t_i;typ="lin",ex_ends=false)
    if typ == "lin" D = LinearBasis(X,t,t_i);  Phi = D[1];  t = D[2];  X = D[3] elseif typ == "con" D = ExtendPoints(X,t,t_i,typ=typ)
        X = D[1];  t = D[2];  i2 = 1;  Phi = zeros(size(t,1),size(t_i,1));  for i in 1:size(t,1) if [t_i; Inf][i2+1] <= t[i] 
        i2 += 1 end;  Phi[i,i2] = 1 end end
    if typ == "lin" Chi = inv(InnerL2(Phi,Phi,t,typ=typ));  Gamma = InnerL2(Phi,X,t,typ=typ)
    elseif typ == "con" Chi = diagm(diag(Phi'*Phi).^(-1));  Gamma = Phi'*X end
    R = zeros(2+(typ=="con"),size(Gamma,1));  R[1,1] = 1;  R[2,size(Gamma,1)] = 1;  if typ=="con" R[3,size(Gamma,1)-1] = 1 end
    if ex_ends return Chi*Gamma-Chi*R'*inv(R*Chi*R')*(R*Chi*Gamma-[X[1]; ones(1+(typ=="con"))*X[size(X,1)]])
    else return Chi*Gamma end
end

function GenerateWarp(Cov,t,Dir,t_i,n;mu=0,varX=0,varH=0,BX=nothing,BH=nothing)
    X = GeneratePCA(Cov,t,n;mu=mu,var=varX,B=BX);
    h = GeneratePCA(Dir,t_i,n;mu=t_i,var=varH,B=BH);
    Y = zeros(size(X))
    for i in 1:n
        D = ApplyWarp(X[:,i],t,h[:,i],t_i);  Y[:,i] = TruncateFunction(D[1],D[2],t)
    end
    return [Y,X,h]
end

function FirstOrderWarpFit(Y,t,t_i,d;typ="lin",tol=0.001,Ter=100)
    X = reshape(Y[:],size(Y));  D = LinearBasis(X,t,t_i,ex_ends=true);  Phi = D[1];  t = D[2]
    w = size(Y,2);  ter = 0;  converge = -ones(size(Y,2));  H = zeros(Ter*size(t_i,1),size(Y,2))
    H[2,:] = ones(size(Y,2));  H_t = reshape(H[:],size(H));  H_ind = trunc.(Int,2*ones(size(Y,2)))
    while (w > 0) & (ter < Ter) X_fit = FitPCA(X,t,d=d)
        for i in 1:size(Y,2)
            if converge[i] != -1 continue end;  h = t_i + [0; FirstOrderWarp(X[:,i],X_fit[:,i],t,Phi,t_i); 0]
            if InnerL2(h-t_i,h-t_i,t_i)[1] < tol^2 converge[i] = ter; w -= 1 end
            E = ApplyWarp(H[1:H_ind[i],i],H_t[1:H_ind[i],i],h,t_i);  H_ind[i] = size(E[1],1);
            H[1:H_ind[i],i] = E[1];  H_t[1:H_ind[i],i] = E[2]
            D = ApplyWarp(X[:,i],t,h,t_i,typ=typ);  X[:,i] = TruncateFunction(D[1],D[2],t,ex_ends=true,typ=typ)
        end;  ter += 1;  print(ter);  print(", ");  println(sum(converge.==-1)) end
    h = zeros(size(t,1),size(Y,2))
    for i in 1:size(Y,2) H_t[H_ind[i],i] = 1;  h[:,i] = EvaluateFunction(H[1:H_ind[i],i],H_t[1:H_ind[i],i],t) end
    return [X,h,H,H_t,converge]
end

function ReadXML(day,month,year;data_size=500,Max=3000,typ="OFF")
    s = "MGPDomandaOfferta.xml"
    Raw_Data = root(parse_file("E:/Data/XML/$year$month$day$s"))["DomandaOfferta"]
    Data_X = zeros(data_size,24);  Data_t = zeros(data_size,24);  Data_ind = trunc.(Int,ones(24));  ind1 = 1;  ind2 = 1
    for i in 1:size(Raw_Data,1)+1
        if ind2 != (i<=size(Raw_Data,1))*parse(Int64, content(find_element(Raw_Data[i-(i>size(Raw_Data,1))],"Ora")))
            if typ == "OFF" Data_t[1:ind1,ind2] = Data_t[ind1,ind2] .- Data_t[ind1:-1:1,ind2]
                Data_X[1:ind1,ind2] = [Data_X[ind1-1:-1:1,ind2]; Data_X[1,ind2]]
            elseif typ == "BID" Data_t[1:ind1,ind2] = [Data_t[1,ind2]; Data_t[ind1,ind2] .+ Data_t[1:ind1-1,ind2]]
                Data_X[1:ind1-(ind1>1),ind2] = [Data_X[ind1-(ind1>1),ind2]; Data_X[1:ind1-1-*(ind1>1),ind2]] end
            Data_ind[ind2] = ind1;  if i>size(Raw_Data,1) break end
            ind2 = parse(Int64, content(find_element(Raw_Data[i],"Ora")));  ind1 = 1 end
        if (content(find_element(Raw_Data[i],"Tipo")) != typ) | (content(find_element(Raw_Data[i],"ZonaMercato")) != "CNOR;CSUD;NORD;SARD;SICI;SUD;AUST;COAC;CORS;FRAN;GREC;ROSN;SLOV;SVIZ;MALT;") continue end
        Data_X[ind1,ind2] = min(parse(Float64, content(find_element(Raw_Data[i],"Prezzo"))),Max)
        Data_t[ind1+1,ind2] = Data_t[ind1,ind2] + parse(Float64, content(find_element(Raw_Data[i],"Quantita")))
        ind1 += 1 end
    return [Data_X,Data_t,Data_ind]
end

function XMLtoCSV(d,M,y;typ="OFF")
    X = ReadXML(d,M,y,data_size=500,Max=600,typ=typ);  C_Mat = [X[1] X[2] [X[3]; zeros(500-24)]]
    C_Mat = C_Mat[:,[1 .+ (0:47).÷2 .+ 24*((0:47).%2); 49]]
    df = DataFrame(C_Mat[1:maximum(X[3]),:], :auto);  if typ=="OFF" s = "MGPDomandaOfferta.csv" else s = "MGPDomanda.csv" end
    CSV.write("E:/Data/CSV/$y$M$d$s", df)
end

function ConvertXMLtoCSV(m,typ="OFF")
    months = [31; 28; 31; 30; 31; 30; 31; 31; 30; 31; 30; 31]
    for M in m for d in 1:months[M]
        XMLtoCSV("0$d"[(d>9).+(1:2)],"0$M"[(M>9).+(1:2)],2019,typ=typ)
    end end
end

function ReadCSV(d,M,y;data_size=500,typ="OFF")
    if typ=="OFF" s = "MGPDomandaOfferta.csv" else s = "MGPDomanda.csv" end
    C = Matrix(CSV.read("E:/Data/CSV/$y$M$d$s",DataFrame))
    C = [C[1:min(size(C,1),data_size),:]; [zeros(max(data_size-size(C,1),0),48) ones(max(data_size-size(C,1),0))]]
    return [C[:,1:2:48], C[:,2:2:48], trunc.(Int,C[1:24,49])]
end

function GetData(m;data_size=500,typ="OFF")
    months = [31; 28; 31; 30; 31; 30; 31; 31; 30; 31; 30; 31];  Data_X = zeros(data_size,24*sum(months[m]))
    Data_t = zeros(data_size,24*sum(months[m]));  Data_ind = trunc.(Int,zeros(24*sum(months[m])))
    day = 1;  i2 = 0;  Ind_Hour = zeros(24*sum(months[m]))
    for M in m for d in 1:months[M]
            D = ReadCSV("0$d"[(d>9).+(1:2)],"0$M"[(M>9).+(1:2)],2019,data_size=data_size,typ=typ);  ind = (24*(day-1)+1):24*day
            Data_X[:,ind] = D[1];  Data_t[:,ind] = D[2];  Data_ind[ind] = D[3];  day += 1
        end end
    for i in 1:size(Data_ind,1)
        if Data_ind[i] == 1 continue end;  i2 += 1;  Ind_Hour[i2] = i
        Data_X[:,i2] = Data_X[:,i];  Data_t[:,i2] = Data_t[:,i];  Data_ind[i2] = Data_ind[i] end
    if typ == "OFF"
        t_max = -Inf;  ind_max = 0
        for i in 1:i2 if t_max < Data_t[Data_ind[i]-1,i] t_max = Data_t[Data_ind[i]-1,i];  ind_max = i end end
        for i in 1:i2 Data_t[Data_ind[i],i] = t_max+1000 end end
    return [Data_X[:,1:i2], Data_t[:,1:i2], Data_ind[1:i2], i2, Ind_Hour[1:i2]]
end

function SortT(Data_t,Data_X,Data_ind,i2;amount = nothing)
    if amount === nothing Sample = 1:i2 else 
        Sample = [1; 2:i2];  for i in 1:amount temp = Sample[i];  ind = trunc.(Int,i2*rand(1)[1])+1
        Sample[i] = Sample[ind];  Sample[ind] = temp end;  Sample = Sample[1:amount] end
    t_sorted = zeros(sum(Data_ind[Sample]));  sorting_ind = trunc.(Int,ones(i2));  w = zeros(sum(Data_ind[Sample]));  ind = 1
    while ind <= size(t_sorted,1)
        t_min = Inf;  ind_min = 0
        for i in Sample
            if sorting_ind[i] > Data_ind[i] continue end
            if t_min > Data_t[sorting_ind[i],i] t_min = Data_t[sorting_ind[i],i];  ind_min = i end
        end
        t_sorted[ind] = t_min
        w[ind] = (Data_X[sorting_ind[ind_min],ind_min] - (sorting_ind[ind_min]>1)*Data_X[max(sorting_ind[ind_min]-1,1),ind_min])/(max(t_min,0.001) - (sorting_ind[ind_min]>1)*Data_t[max(sorting_ind[ind_min]-1,1),ind_min])
        sorting_ind[ind_min] += 1;  ind += 1
    end
    return [t_sorted, w]
end

function ClusterWeighted(t_sorted,Clu,W;Ter=10)
    Clu_mean = zeros(Clu);  ind = 1;  inc = 0
    for i in 1:size(t_sorted,1)
        inc += Clu*W[i]; if inc > ind-0.51 Clu_mean[ind] = t_sorted[i];  ind += 1;  end end
    Clu_mean_old = zeros(Clu);  ter = 1
    while (sum((Clu_mean - Clu_mean_old).^2) < 1) | ter < Ter
        i = 1;  i2 = 1;  ind = 1
        while i <= size(t_sorted,1)
            while (abs(t_sorted[i2]-Clu_mean[ind]) + 0.01) < abs(t_sorted[i2]-Clu_mean[ind+1]) i2 += 1 end
            Clu_mean[ind] = sum(t_sorted[i:i2])/(i2-i+1);  i2 += 1;  i = i2;  ind += 1
            if ind == Clu Clu_mean[ind] = sum(t_sorted[i:size(t_sorted,1)])/(size(t_sorted,1)-i+1);  break end
        end;  ter += 1 end
    Clu_mean[1] = t_sorted[1];  Clu_mean[Clu] = t_sorted[size(t_sorted,1)]
    return Clu_mean
end

function Cluster(t_sorted,Clu;Ter=10)
    Clu_mean = zeros(Clu)
    for i in 1:Clu Clu_mean[i] = t_sorted[((2*i-1)*size(t_sorted,1))÷(2*Clu)] end
    Clu_mean_old = zeros(Clu);  ter = 1
    while (sum((Clu_mean - Clu_mean_old).^2) < 1) | ter < Ter
        i = 1;  i2 = 1;  ind = 1
        while i <= size(t_sorted,1)
            while (abs(t_sorted[i2]-Clu_mean[ind]) + 0.01) < abs(t_sorted[i2]-Clu_mean[ind+1]) i2 += 1 end
            Clu_mean[ind] = sum(t_sorted[i:i2])/(i2-i+1);  i2 += 1;  i = i2;  ind += 1
            if ind == Clu Clu_mean[ind] = sum(t_sorted[i:size(t_sorted,1)])/(size(t_sorted,1)-i+1);  break end
        end;  ter += 1 end
    Clu_mean[1] = t_sorted[1];  Clu_mean[Clu] = t_sorted[size(t_sorted,1)]
    return Clu_mean
end

function TruncateData(Data_X,Data_t,Data_ind,t,i2;typ="con",Max=3000,ex_ends=true)
    Data_Xtil = zeros(size(t,1),i2)
    for i in 1:i2
        Data_Xtil[:,i] = TruncateFunction(Data_X[1:Data_ind[i],i],Data_t[1:Data_ind[i],i],t,typ=typ,ex_ends=ex_ends)
        for j in 2:size(t,1)
            if Data_Xtil[j,i] < Data_Xtil[j-1,i] Data_Xtil[j,i] = (Data_Xtil[j+(j<size(t,1)),i] + 4*Data_Xtil[j-1,i])/5 end
            if Data_Xtil[j,i] > Max Data_Xtil[j,i] = Max end
        end
        print(i);  print(", ") end
    return Data_Xtil
end

function GetSeasonInd(Inds,m,hour=false;variable=false)
    Seasoninds = zeros(size(Inds,1)); n = size(Inds,1)*variable
    if !hour months = [31; 28; 31; 30; 31; 30; 31; 31; 30; 31; 30; 31];  s = 24*sum(months[1:m-1]);  e = 24*sum(months[1:m]) end
    for i in 1:size(Inds,1) if hour if mod(Inds[i],24) == mod(m,24) if variable Seasoninds[i] = 1
        else n += 1;  Seasoninds[n] = i end end elseif (Inds[i] > s) & (Inds[i] <= e)
            if variable Seasoninds[i] = 1 else n += 1;  Seasoninds[n] = i end end end
    return trunc.(Int,Seasoninds[1:n])
end

function TestSeasonality(Scores,Inds,hour=false)
    n = 12*(1+hour);  K = size(Scores,1);  K2 = size(Scores,2);  Means = zeros(K,n);  Vars = zeros(K,n); Ks = zeros(n)
    Mean = Scores*ones(K2)/K2;  Var = ((Scores-Mean*ones(1,K2)).^2)*ones(K2)/(K2-1)
    for i in 1:n SeasonInds = GetSeasonInd(Inds,i,hour);  k = size(SeasonInds,1);  if k == 0 continue end; Ks[i] = k
        Means[:,i] = Scores[:,SeasonInds]*ones(k)/k;  Vars[:,i] = ((Scores[:,SeasonInds]-Means[:,i]*ones(1,k)).^2)*ones(k)/(k-1)
    end
    Stat = zeros(K);  for i in 1:n Stat += Ks[i]*(Means[:,i]-Mean).^2 end;  Stat = Stat./Var
    Test = zeros(K);  for i in 1:K Test[i] = (Stat[i] < quantile.(Chisq(n),0.95)) end
    println(Ks)
    return [Stat Test]
end

function CorrectSeasonality(Scores,Inds,m=0,hours=true,pred=0)
    X = zeros(23*hours+m,size(Inds,1));  Tra = 1:size(Inds,1)-pred;  Pred = size(Inds,1)-pred+1:size(Inds,1)
    for i in 1:m X[i,:] = GetSeasonInd(Inds,i,false,variable=true) end
    if hours;  for i in 2:24 X[m+i-1,:] = GetSeasonInd(Inds,i,true,variable=true) end end
    return [Scores*(I-X[:,Tra]'*inv(X[:,Tra]*X[:,Tra]')*X[:,Tra]), Scores*X[:,Tra]'*inv(X[:,Tra]*X[:,Tra]'), X[:,Pred]]
end

function ForecastVARPCA(X,t,Y,d;Ind_X=1:size(X,2),Ind_Y=size(X,2)+1:size(X,2)+size(Y,2),typ="lin",season=false,R=zeros(size(X,1),size(X,1)))
    P = zeros(size(Y));  error = zeros(size(Y,2))
    for i in 1:size(Y,2)
        if [Ind_X[size(X,2)]; Ind_Y][i]+1 != [Ind_X[size(X,2)]; Ind_Y][i+1] continue end
        print([Ind_X[size(X,2)]; Ind_Y][i:i+1]);  print(", ")
        D = FitPCA([X Y[:,1:i-1]],t,d=d,typ=typ,coefs=true);  Scores = D[1];  Eig = D[2];  Mean = D[3]
        if season E = CorrectSeasonality(Scores,[Ind_X; Ind_Y[1:i]],3,true,1);  Scores = E[1] else E = [0; 0; 0] end
        V = FitVAR(Scores,Ind=[Ind_X; Ind_Y[1:i-1]],R=R);  A = V[1];  P[:,i] = Mean+Eig*(A*Scores[:,size(X,2)+i-1].+season*E[2]*E[3])
        error[i] = InnerL2(P[:,i]-Y[:,i],P[:,i]-Y[:,i],t,typ=typ)[1] end
    return [P, error]
end

function ForecastVARWARPCA(X,t,Y,d,w;Ind_X=1:size(X,2),Ind_Y=size(X,2)+1:size(X,2)+size(Y,2),typ="lin",season=false,R=zeros(size(X,1),size(X,1)))
    P = zeros(size(Y));  H = zeros(size(Y));  error = zeros(size(Y,2))
    for i in 1:size(Y,2)
        if [Ind_X[size(X,2)]; Ind_Y][i]+1 != [Ind_X[size(X,2)]; Ind_Y][i+1] continue end
        print([Ind_X[size(X,2)]; Ind_Y][i:i+1]);  print(", ")
        t_i = [t[1:15:size(t,1)]; t[size(t,1)]];  W = FirstOrderWarpFit([X Y[:,1:i-1]],t,t_i,d,typ=typ,tol=0.001,Ter=20)
        D = FitPCA(W[1],t,d=d,typ=typ,coefs=true);  DW = FitPCA(W[2].-t,t,d=w,typ="lin",with_mu=false,coefs=true)
        Scores = [D[1]; DW[1]];  Eig = [D[2] DW[2]];  Mean = D[3]
        if season E = CorrectSeasonality(Scores,[Ind_X; Ind_Y[1:i]],3,true,1);  Scores = E[1] else E = [0; 0; 0] end
        V = FitVAR(Scores,Ind=[Ind_X; Ind_Y[1:i-1]],R=R);  A = V[1];  P[:,i] = Mean + Eig[:,1:d]*(A[1:d,:]*Scores[:,size(X,2)+i-1] .+ season*E[2][1:d,:]*E[3])
        H[:,i] = t + Eig[:,d+1:d+w]*(A[d+1:d+w,:]*Scores[:,size(X,2)+i-1] .+ season*E[2][d+1:d+w,:]*E[3])
        error[i] = InnerL2(P[:,i]-Y[:,i],P[:,i]-Y[:,i],t,typ=typ)[1] end
    return [P, H, error]
end

function EvaluateFunction(X,t,t_i;typ="lin")
    X_i = zeros(size(t_i));  j = 2
    for i in 1:size(t_i,1) while t_i[i] > t[j] j += 1 end
        if typ=="lin" X_i[i] = X[j-1] + (t_i[i]-t[j-1])*(X[j]-X[j-1])/(t[j]-t[j-1])
        elseif typ=="con" X_i[i] = X[j-1] end end
    return X_i
end

function PiecewiseIntersect(X,t_x,Y,t_y,typ="con")
    D_x = ExtendPoints(X,t_x,t_y,typ=typ);  Xs = D_x[1];  t_xs = D_x[2]
    D_y = ExtendPoints(Y,t_y,t_x,typ=typ);  Ys = D_y[1];  t_ys = D_y[2]
    int = [NaN; NaN] 
    for i in 1:size(t_xs,1)-1
        #print(Xs[i]);  print(", ");  print(Ys[i]);  print("), (")
        if ((Xs[i] > Ys[i]) & (Xs[i+1] < Ys[i+1])) | ((Xs[i] < Ys[i]) & (Xs[i+1] > Ys[i+1]))
            int = [t_xs[i+1]; Xs[i]];  break
        elseif Xs[i] == Ys[i] int = [t_xs[i]; Xs[i]];  break end end
    return int
end

function DemandIntersect(X,t_x,Y,t_y;Ind_x=trunc.(Int,size(X,1)*ones(size(X,2))),Ind_y=trunc.(Int,size(Y,1)*ones(size(Y,2))))
    int = zeros(2,size(Ind_x,1))
    for i in 1:size(Ind_x,1)  int[:,i] = PiecewiseIntersect(X[1:Ind_x[i],i],t_x[1:Ind_x[i],i],Y[1:Ind_y[i],i],t_y[1:Ind_y[i],i]) end
    return int
end

function QQplot(x,y,axes;title="QQ-Plot",save=false)
    scatter(sort(y),sort(x[:,1]),legend = :topleft, label = "x1")
    for i in 2:size(x,2) scatter!(sort(y),sort(x[:,i]),legend = :topleft, label = "x$i") end
    plot!([-100,100],[-100,100],legend = :topleft,xguide=axes[2],yguide=axes[1], title = title, label = "Identity", xlims = [minimum(x),maximum(x)], ylims = [minimum(y),maximum(y)])
    if save != false savefig("E:/Project/_Plots/$save.png") end
end

function Normalize(X)
    X = X-ones(size(X,1),size(X,1))/size(X,1)*X
    return X./sqrt.(ones(1,size(X,1))*(X.^2)/(size(X,1)-1))
end

function PlotPieceCon(x,y;typ="con",legend=false,title=nothing,save=false,label=nothing,axes=nothing)
    X = x;  Y = y;  if typ == "con" X = x[trunc.(Int,((1:2*size(x,1)-2) .+ 2)/2),:];  Y = y[trunc.(Int,((2:2*size(y,1)-1) .+ 2)/2),:] end
    if axes === nothing display(plot(X,Y,legend=legend,title=title,label=label)) else
    display(plot(X,Y,legend=legend,title=title,label=label,xguide=axes[1],yguide=axes[2])) end;  if save != false savefig("E:/Project/_Plots/$save.png") end
end

function DistL2(X,t_x,Y,t_y,typ="con")
    for i in 2:size(t_x,1) if t_x[i] < t_x[i-1] t_x[i] = t_x[i-1] + 0.00001 end end
    for i in 2:size(t_y,1) if t_y[i] < t_y[i-1] t_y[i] = t_y[i-1] + 0.00001 end end
    D_x = ExtendPoints(X,t_x,t_y,typ=typ);  Xs = D_x[1];  t_xs = D_x[2]
    D_y = ExtendPoints(Y,t_y,t_x,typ=typ);  Ys = D_y[1];  t_ys = D_y[2]
    return sqrt(InnerL2(Xs-Ys,Xs-Ys,t_xs,typ=typ)[1])
end

function SampleDistL2(X,t_x,Y,t_y;Ind_x=trunc.(Int,size(X,1)*ones(size(X,2))),Ind_y=trunc.(Int,size(Y,1)*ones(size(Y,2))))
    int = zeros(size(Ind_x,1))
    for i in 1:size(Ind_x,1)  int[i] = DistL2(X[1:Ind_x[i],i],t_x[1:Ind_x[i],i],Y[1:Ind_y[i],i],t_y[1:Ind_y[i],i]) end
    return int
end
