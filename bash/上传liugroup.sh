#上传liugroup
scp -r ./xpkang_tmp/bet.sh xpkang@172.18.31.107:/data/liugroup/home/xpkang/

scp -r ./xpkang_tmp/extract_brain_region_volumn.py xpkang@172.18.31.107:/data/liugroup/home/xpkang/

scp -r ./xpkang_tmp/DMN_region/ xpkang@172.18.31.107:/data/liugroup/home/xpkang/data/DMN_region/

#liugroup下载
scp -r xpkang@172.18.31.107:/data/liugroup/home/xpkang/AD/MCAD/AD_S08/AD_S08_MPR/\*.csv ~/xpkang_tmp/AD/MCAD/AD_S08/AD_S08_MPR/

#查看文件大小
du -sh

#查看完整路径
pwd

#让.sh中的/n/r变成/r（Windows和linux不同点）
sed -i 's/\r$//' <filename>

#后台运行命令
nohup python extract_brain_region_volumn.py --mask_dir ./data/DMN_region --data_dir ./AD/MCAD/AD_S08/AD_S08_MPR &