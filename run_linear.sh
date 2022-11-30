nspp=100
batch=10
numthreads=8
build=build-linear

for scene in {"veach-mis/mis_v4.pbrt","bathroom2/scene-v4.pbrt","cornell-box/scene-v4.pbrt"}
do
    ./${build}/pbrt --spp $nspp --batch $batch --nthreads ${numthreads} --alpha 0.5 --folder ${build}/MIS-Linear ../pbrt-v4-scenes-P3D/${scene}
done
