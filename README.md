# Phase 1
```
python Wrapper.py --videopath ./Data/Data1.mp4 --facepath ./TestSet/Rambo.jpg --twofaces 0 --method Tri
```
`--method`: TPS, Tri, PRNet  
`--videopath`: the video path for Facwswap  
`--facepath`: the image path to replace the face  
`--twofaces`: 1 for True and 0 for False   
Note: 0 for Data1.mp4 and Data2.mp4  
      1 for Data3.mp4 

# Phase 2
```
python Wrapper.py --videopath ./Data/Data1.mp4 --facepath ./TestSet/Rambo.jpg --twofaces 0 --method PRNet
```
`--method`: TPS, Tri, PRNet  
`--videopath`: the video path for Facwswap  
`--facepath`: the image path to replace the face  
`--twofaces`: 1 for True and 0 for False   
Note: 0 for Data1.mp4 and Data2.mp4  
      1 for Data3.mp4