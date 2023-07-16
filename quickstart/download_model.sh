#wget "https://www.dropbox.com/scl/fi/0ffel0b2ug30trjzo08sa/mist_cf_best.ckpt?rlkey=xjlxte1je40dbo5rzsss6avg7"
#
#wget "https://www.dropbox.com/scl/fi/v0qqu8psetcf3g162l62p/fast_filter_best.ckpt?rlkey=gf1danmnud9uy14v9e7cs9c7u"
wget https://zenodo.org/record/8151490/files/fast_filter_best.ckpt
wget https://zenodo.org/record/8151490/files/mist_cf_best.ckpt


mkdir quickstart/models/ 

#mv mist_cf_best.ckpt?rlkey=xjlxte1je40dbo5rzsss6avg7 quickstart/models/mist_cf_best.ckpt
#mv fast_filter_best.ckpt?rlkey=gf1danmnud9uy14v9e7cs9c7u quickstart/models/fast_filter_best.ckpt
mv mist_cf_best.ckpt quickstart/models/mist_cf_best.ckpt
mv fast_filter_best.ckpt quickstart/models/fast_filter_best.ckpt
