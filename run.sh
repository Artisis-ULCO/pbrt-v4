nspp=50
batch=10

./build/pbrt --spp $nspp --batch $batch --nthreads 8 --alpha 0.5 --fixed 1 --folder build/MIS-Equal ../pbrt-v4-scenes-P3D/veach-mis/mis_v4.pbrt
./build/pbrt --spp $nspp --batch $batch --nthreads 8 --alpha 0.99 --fixed 1 --folder build/MIS-BRDF ../pbrt-v4-scenes-P3D/veach-mis/mis_v4.pbrt
./build/pbrt --spp $nspp --batch $batch --nthreads 8 --alpha 0.01 --fixed 1 --folder build/MIS-Light ../pbrt-v4-scenes-P3D/veach-mis/mis_v4.pbrt
./build/pbrt --spp $nspp --batch $batch --nthreads 8 --alpha 0.5 --folder build/MIS-T0-01 --tsallis 0.01 ../pbrt-v4-scenes-P3D/veach-mis/mis_v4.pbrt
./buiil/pbrt --spp $nspp --batch $batch --nthreads 8 --alpha 0.5 --folder build/MIS-T0-5 --tsallis 0.5 ../pbrt-v4-scenes-P3D/veach-mis/mis_v4.pbrt
./build/pbrt --spp $nspp --batch $batch --nthreads 8 --alpha 0.5 --folder build/MIS-T1 --tsallis 1 ../pbrt-v4-scenes-P3D/veach-mis/mis_v4.pbrt
./build/pbrt --spp $nspp --batch $batch --nthreads 8 --alpha 0.5 --folder build/MIS-T1-5 --tsallis 1.5 ../pbrt-v4-scenes-P3D/veach-mis/mis_v4.pbrt
./build/pbrt --spp $nspp --batch $batch --nthreads 8 --alpha 0.5 --folder build/MIS-T2 --tsallis 2 ../pbrt-v4-scenes-P3D/veach-mis/mis_v4.pbrt
