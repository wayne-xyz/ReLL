veichel lidar and DSM , 3Dvs2.5D GICP, DSM only have one z-value per(x,y), v's lidar has many z value per(x,y)

1 chanllenging: 
 modallity mismatch , DSM top envelope , Lidar submap ground view, nearest bias GICP optimize lift the points upward.
unconstrained 6DoF will move to fit roofs or top of trees instead of road

- method1. 
ROI focusing + tighten correspondences+ post -corrections 