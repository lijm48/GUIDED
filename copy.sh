TARGET=/apdcephfs_cq12/share_1150325/jiaminghli/code/OVD/my_LaMI-main_new
SRC=/apdcephfs_cq12/share_1150325/jiaminghli/code/OVD/my_LaMI-main

# 核心代码目录（完整复制）
rsync -av $SRC/tools/ $TARGET/tools/
rsync -av $SRC/lami_dino_moe2/ $TARGET/lami_dino_moe2/
rsync -av $SRC/Mix_data_train_with_vlm_scr/ $TARGET/Mix_data_train_with_vlm_scr/
rsync -av $SRC/FG_OVD_TEST/ $TARGET/FG_OVD_TEST/ --exclude='FG_results/' --exclude='__MACOSX/' --exclude='*.zip'
rsync -av $SRC/Vis_test/ $TARGET/Vis_test/
rsync -av $SRC/configs/ $TARGET/configs/
rsync -av $SRC/utils/ $TARGET/utils/
rsync -av $SRC/detectron2/ $TARGET/detectron2/
rsync -av $SRC/detrex/ $TARGET/detrex/
rsync -av $SRC/dataset/ $TARGET/dataset/ --exclude='FG_OVD/images/'  # 图片数据软链接即可

# 脚本
cp $SRC/train_lvis.sh $TARGET/
cp $SRC/run_mix_data_train_attr.sh $TARGET/
cp $SRC/FG_lami_infer.sh $TARGET/
cp $SRC/setup.py $SRC/setup.cfg $TARGET/
