#!/bin/bash
path='/data/liugroup/AD/MCAD/sMRI'
centers=('/AD_S01/AD_S01_MPR/' '/AD_S02/AD_S02_MPR/' '/AD_S03/AD_S03_MPR/' '/AD_S04/AD_S04_MPR/' '/AD_S05/AD_S05_MPR/' '/AD_S06/AD_S06_MPR/' '/AD_S07/AD_S07_MPR/' '/AD_S08/AD_S08_MPR/')
my_path='/data/liugroup/home/xpkang/AD/MCAD'
nii='.nii'
brain='_brain.nii.gz'
for center in ${centers[@]}
do
    files=$(ls ${path}${center})

    for filename in $files
    do
        if [[ $filename =~ [0-9][0-9][0-9]'.nii' ]]
        then
            bet ${path}${center}${filename} ${my_path}${center}${filename}_brain  -f 0.5 -g 0
        fi
    done
done



