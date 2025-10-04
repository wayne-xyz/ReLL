veichel lidar and DSM , 3Dvs2.5D GICP, DSM only have one z-value per(x,y), v's lidar has many z value per(x,y)

1 chanllenging: 
 modallity mismatch , DSM top envelope , Lidar submap ground view, nearest bias GICP optimize lift the points upward.
unconstrained 6DoF will move to fit roofs or top of trees instead of road

1.1 record , analysis
current analysis method , core, op1, op2 faild , using the 100x100m same area align the lidar point data and the dsm point data , can not align with them, 
- consideration1 , same area 100x100 the lidar data is not even , but the dsm is even cover the arae
- density, 
- utm translate scale the transfor matrix



1.2. record . gicp-veri
minimal verify using one dataset + offset , apply the gicp , workswell 
in folder  , this is basic alignment clue for next 



1.3 record  gicp-veri-lidar-dsm

using the lidar shape to do the align , crop the dsm to same like the lidar shape . using the range 0.5m(within the this range there should be points in the lidar which gain the lidar 3d shape dsm ) get the extraced dsm poiints to do alignment , this is ireegular data align which gain better than the mismatch 3d shape align 
also need the shift before alignemnt 

Good result for the 1.3 record . this method align the shape of the points area , 