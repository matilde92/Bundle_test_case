clear all
close all

load stats_forced.dat
%load stats_mixed.dat % for mixed convection

PoD = 1.4;
phi = 2*sqrt(3)/pi*PoD^2-1;
R = 1/2/phi;
Ph = PoD*R;

nx=256;
ny=448;


count = 0;

for i=1:ny
    for j=1:nx
        count=count+1;
        x(i,j)=stats(count,1);
        y(i,j)=stats(count,2);
        u(i,j)=stats(count,3);
        v(i,j)=stats(count,4);
        w(i,j)=stats(count,5);
        T(i,j)=stats(count,6);
        uu(i,j)=stats(count,7);
        uv(i,j)=stats(count,8);
        uw(i,j)=stats(count,9);
        vv(i,j)=stats(count,10);
        vw(i,j)=stats(count,11);
        ww(i,j)=stats(count,12);
        uT(i,j)=stats(count,13);
        vT(i,j)=stats(count,14);
        wT(i,j)=stats(count,15);
        TT(i,j)=stats(count,16);
        dudx(i,j)=stats(count,17);
        dudy(i,j)=stats(count,18);
        dudz(i,j)=stats(count,19);
        dvdx(i,j)=stats(count,20);
        dvdy(i,j)=stats(count,21);
        dvdz(i,j)=stats(count,22);
        dwdx(i,j)=stats(count,23);
        dwdy(i,j)=stats(count,24);
        dwdz(i,j)=stats(count,25);
        dTdx(i,j)=stats(count,26);
        dTdy(i,j)=stats(count,27);
        dTdz(i,j)=stats(count,28);
        uxux(i,j)=stats(count,29);
        uyuy(i,j)=stats(count,30);
        uzuz(i,j)=stats(count,31);
        vxvx(i,j)=stats(count,32);
        vyvy(i,j)=stats(count,33);
        vzvz(i,j)=stats(count,34);
        wxwx(i,j)=stats(count,35);
        wywy(i,j)=stats(count,36);
        wzwz(i,j)=stats(count,37);
        TxTx(i,j)=stats(count,38);
        TyTy(i,j)=stats(count,39);
        TzTz(i,j)=stats(count,40);
    end
end

k = 0.5*(uu + vv + ww);
epsilon = (uxux+uyuy+uzuz+vxvx+vyvy+vzvz+wxwx+wywy+wzwz);
kT = 0.5*TT;
epsilonT = (TxTx+TyTy+TzTz);

