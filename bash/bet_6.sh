#!/bin/bash
path='/data/liugroup/AD/MCAD/sMRI'
centers=('/AD_S06/AD_S06_MPR/')
my_path='/data/liugroup/home/xpkang/AD/MCAD'
nii='.nii'
brain='_brain.nii.gz'
for center in ${centers[@]}
do
    files=$(ls ${path}${center})
    echo 
    for filename in $files
    do
        if [[ $filename =~ 'MPR.nii' ]]
        then
            bet ${path}${center}${filename} ${my_path}${center}${filename}_brain  -f 0.5 -g 0
        fi
    done
done



