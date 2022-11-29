nspp=100
batch=10
buildFolder=build

#./${buildFolder}/pbrt --spp $nspp --batch $batch --nthreads 8 --alpha 0.5 --fixed 1 --folder ${buildFolder}/MIS-Equal ../pbrt-v4-scenes-P3D/veach-mis/mis_v4.pbrt
#./${buildFolder}/pbrt --spp $nspp --batch $batch --nthreads 8 --alpha 0.99 --fixed 1 --folder ${buildFolder}/MIS-BRDF ../pbrt-v4-scenes-P3D/veach-mis/mis_v4.pbrt
#./${buildFolder}/pbrt --spp $nspp --batch $batch --nthreads 8 --alpha 0.01 --fixed 1 --folder ${buildFolder}/MIS-Light ../pbrt-v4-scenes-P3D/veach-mis/mis_v4.pbrt
./${buildFolder}/pbrt --spp $nspp --batch $batch --nthreads 8 --alpha 0.5 --folder ${buildFolder}/MIS-Linear ../pbrt-v4-scenes-P3D/veach-mis/mis_v4.pbrt
