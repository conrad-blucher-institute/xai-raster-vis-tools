

FRAME_DIR="frames"


for i in {1..384}
do
    python getTopBands.py \
        -b $i \
        -i $FRAME_DIR""/shap_2019_color0.5_$i"".png \
        -p shap_values_new/shap_fog_2019_color0.5.pickle \
        --pickled_shap_2 shap_values_new/shap_non_2019_color0.5.pickle \
        -n fog \
        --class_name_2 non-fog \
        --groups 108,204,312,372
done
