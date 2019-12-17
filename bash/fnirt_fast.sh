#!/bin/bash
centers=('/AD_S01/AD_S01_MPR/' '/AD_S02/AD_S02_MPR/' '/AD_S03/AD_S03_MPR/' '/AD_S04/AD_S04_MPR/' '/AD_S05/AD_S05_MPR/' '/AD_S06/AD_S06_MPR/' '/AD_S07/AD_S07_MPR/' '/AD_S08/AD_S08_MPR/')
my_path='/data/liugroup/home/xpkang/AD/MCAD'

brain='_brain.nii.gz'
flirt='_flirt.nii.gz'
warpcoef='_flirt_warpcoef.nii.gz'
fnirt='_flirt_fnirt.nii.gz'
for center in ${centers[@]}
do
    files=$(ls ${my_path}${center})

    for filename in $files
    do
        if [[ $filename =~ $flirt ]]
        then
            fnirt --in=${my_path}${center}${filename} --ref=/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz
            echo ${my_path}${center}${filename} >> log.txt
        fi
    done

    files=$(ls ${my_path}${center})

    for filename in $files
    do
        if [[ $filename =~ $flirt ]]
        then
            applywarp -r /usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz -i ${my_path}${center}${filename} -w ${my_path}${center}${filename/%${flirt}/${warpcoef}} -o ${my_path}${center}${filename/%${flirt}/${fnirt}}
            echo ${my_path}${center}${filename} >> log.txt
        fi
    done

    files=$(ls ${my_path}${center})

    for filename in $files
    do
        if [[ $filename =~ $fnirt ]]
        then
            fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -o ${my_path}${center}${filename} ${my_path}${center}${filename}
            echo ${my_path}${center}${filename} >> log.txt
        fi
    done
done