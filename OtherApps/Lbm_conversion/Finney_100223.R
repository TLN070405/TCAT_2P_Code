library(ggplot2)
require(mgcv)
require(data.table)
require(plot3D)
require(RColorBrewer)
require(scatterplot3d)

ReadTwoPhaseAverages<-function(PATH){
  # Read two phase averages from 'timelog.tcat' file and non-dimensionalize
  # simluation input and output files must be present at PATH
  #    Color.in
  #    Domain.in
  #    timelog.tcat
  
  # Read the input parameters
  FILENAME=paste0(PATH,"/Color.in")
  PRM<-scan(file=FILENAME)        
  IFT<-PRM[5]*5.796
  VISC<-(PRM[1]-0.5)/3
  NU1<-(PRM[1]-0.5)/3
  NU2<-(PRM[2]-0.5)/3
  RHO1<-PRM[3]
  RHO2<-PRM[4]
  Fx<-PRM[7]
  Fy<-PRM[8]
  Fz<-PRM[9]
  PI=3.14159265358979
  
  FILENAME=paste0(PATH,"/Domain.in")
  DM<-scan(file=FILENAME)        
  Px<-DM[1]
  Py<-DM[2]
  Pz<-DM[3]
  Nx<-DM[4]
  Ny<-DM[5]
  Nz<-DM[6]
  
  NX=Px*Nx
  NY=Py*Ny
  NZ=Pz*Nz
  
  VOLUME=NX*NY*NZ
  
  # Read the time history of averages 
  FILENAME=paste0(PATH,"/timelog.tcat")
  DATA<-read.csv(file=FILENAME,head=TRUE,sep=" ")
  
  # Compute the Sauter mean for non-dimensionalization
  D32=120
  
  # interfacial tension
  DATA$IFT<-IFT
  # dynamic viscosity
  DATA$viscosity<-VISC
  # external force
  DATA$Fx<-Fx
  DATA$Fy<-Fy
  DATA$Fz<-Fz
  
  #	DATA$Euler<-DATA$Euler*D32^3/VOLUME
  #	DATA$An<-DATA$An*D32/VOLUME
  #	DATA$pwn<-DATA$trJwn*D32
  #	DATA$awn<-DATA$awn*(-1)
  #	DATA$aws<-DATA$aws*(-1)*D32
  #	DATA$ans<-DATA$ans*(-1)*D32
  
  DATA$Source<-PATH
  
  return(DATA)
}


setwd("LOGS")

# Read in the global averages from steady drainage
LIST<-list.files(".",pattern="drain_")

for (i in LIST){
  print(i)
  D<-ReadTwoPhaseAverages(i)
  D<-subset(D,D$time>75000)
  assign(i,D)
}
data<-lapply(ls(pattern="Finney_drain_"),get)
finney_drain<-data.frame(rbindlist(data,use.names=TRUE))

LIST<-list.files(".",pattern="imb1_")

for (i in LIST){
  print(i)
  D<-ReadTwoPhaseAverages(i)
  D<-subset(D,D$time>10000)
  assign(i,D)
}
data<-lapply(ls(pattern="Finney_imb1_"),get)
finney_imb1<-data.frame(rbindlist(data,use.names=TRUE))

LIST<-list.files(".",pattern="drain2_")

for (i in LIST){
  print(i)
  D<-ReadTwoPhaseAverages(i)
  D<-subset(D,D$time>10000)
  assign(i,D)
}
data<-lapply(ls(pattern="Finney_drain2_"),get)
finney_drain2<-data.frame(rbindlist(data,use.names=TRUE))

setwd("../Plots")
finney_drain$Process<-"Primary Drainage"
finney_drain2$Process<-"Secondary Drainage"
finney_imb1$Process<-"Primary Imbibition"

All<-rbind(finney_imb1,finney_drain,finney_drain2)


setwd("../")
sw=expression(s^{bar(bar(w))})
ewn=expression(epsilon^{"'"}^{bar(bar(wn))})
Xn=expression(chi^{"'"}^{bar(bar(n))})
Jwn=expression(J[w]^{"'wn"})