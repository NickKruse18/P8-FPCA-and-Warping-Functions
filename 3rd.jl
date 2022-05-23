include("2nd.jl")

A = [0.8 0 0 0.5 0; 0 0.75 0 0 0.5; 0 0 0.75 0 0; 0 0 0 0.6 0; 0 0 0 0 0.6];  n = 1000
Cov = randn(100,3);  Cov = eigvecs(eigen(Cov*Cov'))[:,100:-1:98]
m = 100;  R1 = trunc.(Int,zeros(25,m));  R2 = trunc.(Int,zeros(25,m))

for i in 1:m
    C = [rand(TDist(5),(1,n)); rand(Chi(1),(1,n)); rand(Normal(0,1),(3,n))];  B = GenerateVAR(A,n,B=C)
    X = GeneratePCA(Cov,1:100,n,var=1,B=B[1:3,:],E=rand(TDist(5),(size(Cov,1),n))*0.0316)
    D = FitPCA(X,1:100,d=3,coefs=true)
    Ind = [1; 2; 3; 4; 5]
    for i in 1:3
        for j in 1:3
            if abs.(D[2]'*Cov)[i,j]==maximum(abs.(D[2]'*Cov)[i,:]) Ind[i] = j end
        end
    end
    
    Ind = [trunc.(Int,abs.(D[2]'*Cov).+0.2)*[1; 2; 3]; [4; 5]]
    print(D[2]'*Cov); print(", ")
    V = FitVAR([D[1]; B[4:5,:]])
    Boot = BootStrapVAR(V[2],[D[1]; B[4:5,:]]-V[2],V[1],1000)
    R1[:,i] = trunc.(Int,ConfidenceInterval(Boot,5,5,[0.025; 0.975])[1])[Ind,Ind][:]

    Boot = BootStrapPCAWARVAR(FitPCA(X,1:100,d=3),X-FitPCA(X,1:100,d=3),D[2],0,0,0,1:100,100,10,B=B[4:5,:],typ="con")
    R2[:,i] = trunc.(Int,ConfidenceInterval(Boot,5,5,[0.025; 0.975])[1])[Ind,Ind][:]
end

reshape(((R1-(A.==0)[:]*ones(1,m)).^2)*ones(m),5,5)
reshape(((R2-(A.==0)[:]*ones(1,m)).^2)*ones(m),5,5)
sum(reshape(((R1-(A.==0)[:]*ones(1,m)).^2)*ones(m),5,5))
sum(reshape(((R2-(A.==0)[:]*ones(1,m)).^2)*ones(m),5,5))


include("2nd.jl")


PlotPieceCon(0:4,0:4,title="Regular case",save="Ind_1")
PlotPieceCon([0; 1; 2.5; 3.5; 4],0:4,title="One generator increases quantity",save="Ind_2")
PlotPieceCon(0:4,0:4,typ="lin",title="h in the regular case",save="Ind_3")
PlotPieceCon([0; 1; 2.5; 3.5; 4],[0; 1; 2; 3; 4],typ="lin",title="h used to simulate increased quantity",save="Ind_4")

B = GetData(1:3,typ="BID")
D = GetData(1:3)
p_high = D[1][D[3][1]];  q_high = D[2][D[3][1]]
B[1] = B[1]/p_high; B[2] = B[2]/q_high
D[1] = D[1]/p_high; D[2] = D[2]/q_high

D[1][2:500,:] = D[1][2:500,:] + (D[1][2:500,:] .== 0)
D[2][2:500,:] = D[2][2:500,:] + (D[2][2:500,:] .== 0)
B[2][2:500,:] = B[2][2:500,:] + (B[2][2:500,:] .== 0)

PlotPieceCon(D[2],D[1], title = "Supply Curves, Jan-Mar 2019", save = "jan_mar_2019",axes=["quantity","price per unit"])
PlotPieceCon(B[2],B[1], title = "Demand Curves, Jan-Mar 2019", save = "jan_mar_2019_demand",axes=["quantity","price per unit"])

#FPCA
Data_X = D[1];  Data_t = D[2];  Data_ind = D[3];  i2 = D[4];  Ind_Hour = D[5]
Sort = SortT(Data_t,Data_X,Data_ind,i2);  t_sorted = Sort[1];  w_sorted = Sort[2].+0.01+0.0001*(1:size(Sort[2],1))
w_sorted = w_sorted/sum(w_sorted)
Clu = 250;  Clu_mean = ClusterWeighted(t_sorted,Clu,w_sorted,Ter=1)
Data_t = Data_t/Clu_mean[Clu];  Clu_mean = Clu_mean/Clu_mean[Clu]
Data_Xtil = TruncateData(Data_X,Data_t,Data_ind,Clu_mean,i2,Max=100000)

d_f = 5
PlotPieceCon(Clu_mean,Data_Xtil, title = "Truncated Supply Curves, Jan-Mar 2019", save = "jan_mar_2019_trun",axes=["quantity","price per unit"])
PlotPieceCon(Clu_mean,FitPCA(Data_Xtil,Clu_mean,d=d_f,coefs=true,typ="con")[3], title = "Truncated Mean, Jan-Mar 2019", save = "jan_mar_2019_trun_mean",axes=["quantity","price per unit"])
PlotPieceCon(Clu_mean,FitPCA(Data_Xtil,Clu_mean,d=d_f,coefs=true,typ="con")[2],legend = :topleft, label = ["e1" "e2" "e3" "e4" "e5"], title = "Eigenfunctions: d = $d_f, Jan-Mar 2019", save = "jan_mar_2019_trun_eig_5",axes=["quantity","price per unit"])
PlotPieceCon(Clu_mean,FitPCA(Data_Xtil,Clu_mean,d=d_f,coefs=false,typ="con"), title = "Fitted Functions: d = $d_f, Jan-Mar 2019", save = "jan_mar_2019_trun_fit_5",axes=["quantity","price per unit"])

#Warping

include("2nd.jl")

t_i = [Clu_mean[1:10:Clu]; Clu_mean[Clu]]
W = FirstOrderWarpFit(Data_Xtil,Clu_mean,t_i,0,typ="con",tol=0.001,Ter=20)
pves = FitPCA(W[2],Clu_mean,d=0,coefs=true)[4][Clu:-1:Clu-9]/sum(FitPCA(W[2],Clu_mean,d=0,coefs=true)[4])
plot!(cumsum(pves),markershape = :diamond,title = "Cumulative pve for each Principal Component: d_h",label="d = 2",legend = :bottomright)
cumsum(pves)
d_f = 4
PlotPieceCon(Clu_mean,W[1]/600, title = "Supply corrected for Warping, Jan-Mar 2019",save = "jan_mar_2019_corr",axes=["quantity","price per unit"])
PlotPieceCon(Clu_mean,W[2],typ="lin",legend = false , title = "Inverted Warping Functions, Jan-Mar 2019",save = "jan_mar_2019_corr_warp")
PlotPieceCon(Clu_mean,FitPCA(W[1]/600,Clu_mean,d=0,coefs=true,typ="con")[3], title = "Corrected Mean Function, Jan-Mar 2019",save = "jan_mar_2019_corr_mean",axes=["quantity","price per unit"])
PlotPieceCon(Clu_mean,FitPCA(W[2].-Clu_mean,Clu_mean,d=d_f,with_mu=false,coefs=true)[2],typ="lin",legend = :topleft, label = ["e1" "e2" "e3" "e4"], title = "Warping Eigenfunctions: d_h = $d_f, Jan-Mar 2019",save = "jan_mar_2019_corr_eig_4")
PlotPieceCon(Clu_mean,FitPCA(W[2].-Clu_mean,Clu_mean,d=d_f,with_mu=false,coefs=false).+Clu_mean,typ="lin", title = "Fitted Inverted Warpings: d_h = $d_f, Jan-Mar 2019",save = "jan_mar_2019_warp_fit_4")
PlotPieceCon(FitPCA(W[2].-Clu_mean,Clu_mean,d=d_f,with_mu=false,coefs=false).+Clu_mean,FitPCA(W[1]/600,Clu_mean,d=0,coefs=true,typ="con")[3],typ="con", title = "Fitted Supply Model A: d_h = $d_f, Jan-Mar 2019",save = "jan_mar_2019_corr_fit_4",axes=["quantity","price per unit"])
Scores = FitPCA(W[2].-Clu_mean,Clu_mean,d=d_f,with_mu=false,coefs=true)[1]
histogram(Scores',legend = :topleft, label = ["s1" "s2" "s3" "s4"], title = "Scores for Model A: d_h = $d_f, Jan-Mar 2019")
X = CorrectSeasonality(Scores,Ind_Hour,3,true)[1]
histogram(X',legend = :topleft, label = ["x1" "x2" "x3" "x4"], title = "Scores after correcting for Seasonality")

QQplot(Normalize(X'),quantile.(Normal(0,1),(1:730)/731),["Standardized Residuals","Normal Distribution"],title="QQ-plot of Model A Residuals",save="jan_mar_2019_warp_qq")

sqrt(sum(SampleDistL2(Data_Xtil,Clu_mean*ones(1,730),Data_X,Data_t,Ind_y=Data_ind).^2)/730)/600

V = FitVAR(X,Ind=Ind_Hour)
Boot = BootStrapVAR(V[2],X-V[2],V[1],1000,Ind=Ind_Hour)
X_W = FitPCA(W[2].-Clu_mean,Clu_mean,d=d_f,with_mu=false,coefs=false,typ="con").+Clu_mean;  E_W = FitPCA(W[2].-Clu_mean,Clu_mean,d=d_f,with_mu=false,coefs=true,typ="con")[2]
Boot2 = BootStrapPCAWARVAR(X_W,W[2]-X_W,E_W,0,0,0,Clu_mean,100,10,Ind=Ind_Hour,typ="con",season=true)
Restrict1 = trunc.(Int,ConfidenceInterval(Boot,4,4,[0.025; 0.975])[1])
Restrict2 = trunc.(Int,ConfidenceInterval(Boot2,4,4,[0.025; 0.975])[1])

D[2] = D[2]/q_high;  B[2] = B[2]/q_high

Equal = DemandIntersect(D[1],D[2],B[1],B[2],Ind_x=D[3],Ind_y=B[3])

sqrt.((DemandIntersect(Data_Xtil,Clu_mean*ones(1,730),B[1],B[2],Ind_y=B[3])-Equal).^2*ones(730)/730)

E = GetData(4)
F = GetData(4,typ="BID")
E[2] = E[2]/q_high;  E[5] = E[5] .+ D[5][730];  F[2] = F[2]/q_high;  F[5] = F[5] .+ B[5][730]
E[1] = E[1]/p_high;  E[2] = E[2]/q_high;  E[5] = E[5] .+ D[5][730]
F[1] = F[1]/p_high;  F[2] = F[2]/q_high;  F[5] = F[5] .+ B[5][730]
Data_Ftil = TruncateData(E[1],E[2],E[3],Clu_mean,E[4],Max=100000)
PlotPieceCon(Clu_mean,Data_Ftil, title = "Truncated Supply Curves, Apr 2019",save="apr_2019_trun",axes=["quantity","price per unit"])
Equal = DemandIntersect(E[1],E[2],F[1],F[2],Ind_x=E[3],Ind_y=F[3])

EW = load("E:/Project/dataB.jld")["data"]

EW = ForecastVARWARPCA(Data_Xtil,Clu_mean,Data_Ftil,0,4,Ind_X=Ind_Hour,Ind_Y=E[5],typ="con",season=true)
save("E:/Project/dataBR.jld", "data", EW)
save("E:/Project/dataB.jld", "data", EW)

sqrt(sum(SampleDistL2(EW[1][:,(EW[3].!=0)],EW[2][:,(EW[3].!=0)],E[1][:,(EW[3].!=0)],E[2][:,(EW[3].!=0)],Ind_y=E[3][(EW[3].!=0)]).^2)/En)/600
En = sum(EW[1][2,:].!=0)

sqrt(sum(EW[2])/En)/600

sqrt.((DemandIntersect(EW[1][:,(EW[3].!=0)],EW[2][:,(EW[3].!=0)],F[1][:,(EW[3].!=0)],F[2][:,(EW[3].!=0)],Ind_y=F[3][(EW[3].!=0)])-Equal[:,(EW[3].!=0)]).^2*ones(En)/En)


#Inverse
Data_X = D[2];  Data_t = D[1];  Data_ind = D[3];  i2 = D[4];  Ind_Hour = D[5]
Sort = SortT(Data_t,Data_X,Data_ind,i2);  t_sorted = Sort[1]
CluI = 120;  Clu_meanI = Cluster(t_sorted,CluI,Ter=1)
Data_Xtil = TruncateData(Data_X,Data_t,Data_ind,Clu_meanI,i2,Max=100000,ex_ends=false)
d_f = 4

PlotPieceCon(Clu_meanI,Data_Xtil, title = "Truncated Inverted Supply Curves, Jan-Mar 2019",save="jan_mar_2019_trun_inv",axes=["price per unit","quantity"])
PlotPieceCon(Clu_meanI,FitPCA(Data_Xtil,Clu_meanI,d=d_f,coefs=true,typ="con")[3], title = "Truncated Mean, Jan-Mar 2019",save="jan_mar_2019_inv_mean",axes=["price per unit","quantity"])
PlotPieceCon(Clu_meanI,FitPCA(Data_Xtil,Clu_meanI,d=d_f,coefs=true,typ="con")[2],legend = :topright, label = ["e1" "e2" "e3" "e4"], title = "Eigenfunctions: d = $d_f, Jan-Mar 2019",save="jan_mar_2019_inv_eig_4",axes=["price per unit","quantity"])
PlotPieceCon(Clu_meanI,FitPCA(Data_Xtil,Clu_meanI,d=d_f,coefs=false,typ="con"), title = "Fitted Supply Model B: d = $d_f, Jan-Mar 2019",save="jan_mar_2019_inv_fit_4",axes=["price per unit","quantity"])

include("2nd.jl")

Scores = FitPCA(Data_Xtil,Clu_meanI,d = d_f,coefs=true,typ="con")[1]
histogram(Scores',legend = :topleft, label = ["s1" "s2" "s3" "s4"], title = "Scores for Model B: d = $d_f, Jan-Mar 2019")
X = CorrectSeasonality(Scores,Ind_Hour,3,true)[1]
histogram(X',legend = :topleft, label = ["x1" "x2" "x3" "x4"], title = "Scores after correcting for Seasonality")
QQplot(Normalize(X'),quantile.(Normal(0,1),(1:730)/731),["Standardized Residuals","Normal Distribution"],title="QQ-plot of Model B Residuals",save="jan_mar_2019_inv_qq")

V = FitVAR(X,Ind=Ind_Hour)

Boot = BootStrapVAR(V[2],X-V[2],V[1],1000,Ind=Ind_Hour)

X_P = FitPCA(Data_Xtil,Clu_meanI,d=d_f,coefs=false,typ="con");  E_P = FitPCA(Data_Xtil,Clu_meanI,d=d_f,coefs=true,typ="con")[2]

Boot2 = BootStrapPCAWARVAR(X_P,Data_Xtil-X_P,E_P,0,0,0,Clu_meanI,100,10,Ind=Ind_Hour,typ="con")

Restrict1 = trunc.(Int,ConfidenceInterval(Boot,d_f,d_f,[0.025; 0.975])[1])
Restrict2 = trunc.(Int,ConfidenceInterval(Boot2,d_f,d_f,[0.025; 0.975])[1])

Data_Ftil = TruncateData(E[2],E[1],E[3],Clu_meanI,E[4],Max=100000,ex_ends=false)

include("2nd.jl")

sum(Data_Ftil)

EI = ForecastVARPCA(Data_Xtil,Clu_meanI,Data_Ftil,4,Ind_X=Ind_Hour,Ind_Y=E[5],typ="con",season=true)
sqrt(sum(EI[2])/En)
En
Equal = DemandIntersect(E[1],E[2],F[1],F[2],Ind_x=E[3],Ind_y=F[3])

sqrt.((Equal[:,2:292][:,(E[5][2:292]-E[5][1:291] .== 1)] - Equal[:,1:291][:,(E[5][2:292]-E[5][1:291] .== 1)]).^2*ones(221)/221)
sqrt(sum(SampleDistL2(E[1][:,1:291][:,(E[5][2:292]-E[5][1:291] .== 1)],E[2][:,1:291][:,(E[5][2:292]-E[5][1:291] .== 1)],E[1][:,2:292][:,(E[5][2:292]-E[5][1:291] .== 1)],E[2][:,2:292][:,(E[5][2:292]-E[5][1:291] .== 1)],Ind_x=E[3][1:291][(E[5][2:292]-E[5][1:291] .== 1)],Ind_y=E[3][2:292][(E[5][2:292]-E[5][1:291] .== 1)]).^2)/221)

sqrt(sum(SampleDistL2(Clu_meanI*ones(1,En),EI[1][:,(EI[2].!=0)],Clu_meanI*ones(1,En),Data_Ftil[:,(EI[2].!=0)]).^2)/En)

sqrt(sum(SampleDistL2(Clu_meanI*ones(1,En),EI[1][:,(EI[2].!=0)],E[1][:,(EI[2].!=0)],E[2][:,(EI[2].!=0)],Ind_y=E[3][(EI[2].!=0)]).^2)/En)
sqrt.((DemandIntersect(Clu_meanI*ones(1,En),EI[1][:,(EI[2].!=0)],F[1][:,(EI[2].!=0)],F[2][:,(EI[2].!=0)],Ind_y=F[3][(EI[2].!=0)])-Equal[:,(EI[2].!=0)]).^2*ones(En)/En)

sum((EI[1][2:120,(EI[2].!=0)]-EI[1][1:119,(EI[2].!=0)] .< 0))

E[1][2:500,:] = E[1][2:500,:] + (E[1][2:500,:] .== 0)
E[2][2:500,:] = E[2][2:500,:] + (E[2][2:500,:] .== 0)
F[2][2:500,:] = F[2][2:500,:] + (F[2][2:500,:] .== 0)

PlotPieceCon(E[2],E[1], title = "Supply Curves, Apr 2019", save = "apr_2019",axes=["quantity","price per unit"])
PlotPieceCon(F[2],F[1], title = "Demand Curves, Apr 2019", save = "apr_2019_demand",axes=["quantity","price per unit"])