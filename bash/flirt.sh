in_path='/data/liugroup/home/xpkang/adni/'
out_path='/data/liugroup/home/xpkang/adni_reg/'
files=$(ls ${in_path})

for filename in $files
do
    /data/liugroup/home/xpkang/software/DiffusionKitSetup-x86_64-v1.5-r180928/bin/reg_aladin -flo ${in_path}${filename} -ref /data/liugroup/home/xpkang/templates/ch2.nii -res ${out_path}${filename}
    echo ${in_path}${filename} >> log.txt
done