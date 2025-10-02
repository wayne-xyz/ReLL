veichel lidar and DSM , 3Dvs2.5D GICP, DSM only have one z-value per(x,y), v's lidar has many z value per(x,y)

1 chanllenging: 
 modallity mismatch , DSM top envelope , Lidar submap ground view, nearest bias GICP optimize lift the points upward.
unconstrained 6DoF will move to fit roofs or top of trees instead of road

1.1 record , 
current analysis method , core, op1, op2 faild , using the 100x100m same area align the lidar point data and the dsm point data , can not align with them, 
- consideration1 , same area 100x100 the lidar data is not even , but the dsm is even cover the arae
- density, 
- utm translate scale the transfor matrix



1.2. record .
minimal verify using one dataset + offset , apply the gicp , workswell 
in folder  gicp-veri

1.3 record
using folder 
using smaller size 32x32 
using ground data under 0.5m for both 

1.4 record 
using gicp-veri-lidar-dsm
using the lidar shape to do the align , crop the dsm to same like the lidar shape . recotangle shape 80%points in this shape . 
irregular shape of the sample