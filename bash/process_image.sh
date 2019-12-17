path=$1
files=$(ls $path)
for filename in $files
do
    if [[ $filename == [0-9][0-9][0-9]'.nii' ]]
    then
        bet ${filename} ${filename}_brain  -f 0.5 -g 0
    fi
done

path=$1
files=$(ls $path)
brain='_brain.nii.gz'
for filename in $files
do
    if [[ $filename =~ $brain ]]
    then
        flirt -in ${filename} -ref /usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz -out ${filename}_flirt -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear
    fi
done

path=$1
files=$(ls $path)
flirt='_flirt.nii.gz'
for filename in $files
do
    if [[ $filename =~ $flirt ]]
    then
        fnirt -in ${filename} -ref /usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz
    fi
done

path=$1
files=$(ls $path)
flirt='_flirt.nii.gz'
warpcoef='_flirt_warpcoef.nii.gz'
fnirt='_flirt_fnirt.nii.gz'
for filename in $files
do
    if [[ $filename =~ $flirt ]]
    then
        applywarp -r /usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz -i ${filename} -w ${filename/%${flirt}/${warpcoef}} -o ${filename/%${flirt}/${fnirt}}
    fi
done


path=$1
files=$(ls $path)
fnirt='flirt_fnirt.nii.gz'
for filename in $files
do
    if [[ $filename =~ $fnirt ]]
    then
        fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -o $filename $filename
    fi
done
