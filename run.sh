nspp=100
batch=10
numthreads=8
build=build

#"bathroom2/scene-v4.pbrt","cornell-box/scene-v4.pbrt"}
for scene in {"veach-mis/mis_v4.pbrt","veach-mis/mis_v4_specific.pbrt"}
do
    ./${build}/pbrt --spp $nspp --batch $batch --nthreads ${numthreads} --alpha 0.5 --fixed 1 --folder ${build}/MIS-Equal ../pbrt-v4-scenes-P3D/${scene}
    ./${build}/pbrt --spp $nspp --batch $batch --nthreads ${numthreads} --alpha 0.99 --fixed 1 --folder ${build}/MIS-BRDF ../pbrt-v4-scenes-P3D/${scene}
    ./${build}/pbrt --spp $nspp --batch $batch --nthreads ${numthreads} --alpha 0.01 --fixed 1 --folder ${build}/MIS-Light ../pbrt-v4-scenes-P3D/${scene}
    ./${build}/pbrt --spp $nspp --batch $batch --nthreads ${numthreads} --alpha 0.5 --folder ${build}/MIS-T0-01 --tsallis 0.01 ../pbrt-v4-scenes-P3D/${scene}
    ./${build}/pbrt --spp $nspp --batch $batch --nthreads ${numthreads} --alpha 0.5 --folder ${build}/MIS-T0-5 --tsallis 0.5 ../pbrt-v4-scenes-P3D/${scene}
    ./${build}/pbrt --spp $nspp --batch $batch --nthreads ${numthreads} --alpha 0.5 --folder ${build}/MIS-T1 --tsallis 1 ../pbrt-v4-scenes-P3D/${scene}
    ./${build}/pbrt --spp $nspp --batch $batch --nthreads ${numthreads} --alpha 0.5 --folder ${build}/MIS-T1-5 --tsallis 1.5 ../pbrt-v4-scenes-P3D/${scene}
    ./${build}/pbrt --spp $nspp --batch $batch --nthreads ${numthreads} --alpha 0.5 --folder ${build}/MIS-T2 --tsallis 2 ../pbrt-v4-scenes-P3D/${scene}
done
