
OUTFOLDER_RAW="results/sirius_gnps_pred/sirius_1_raw/"
OUTFOLDER="results/sirius_gnps_pred/sirius_1/"
mkdir -p $OUTFOLDER
mkdir -p $OUTFOLDER_RAW

password=$SIRIUS_PW
INPUT_FILE=data/nist_canopus/split_1_test.mgf
#SIRIUS=../mist-dev/sirius5/sirius/bin/sirius
CORES=32


#$SIRIUS login -u samlg@mit.edu -p
#$SIRIUS  \
#    --cores $CORES \
#    --output  $OUTFOLDER_RAW \
#    --input $INPUT_FILE \
#    formula  \
#    --tree-timeout 120 \
#    --compound-timeout 120 \
#    write-summaries \
#    --output $OUTFOLDER \

$SIRIUS  \
    --cores $CORES \
    --output  $OUTFOLDER_RAW \
    --input $INPUT_FILE \
    formula  \
    -i "[M+H]+,[M+K]+,[M+Na]+,[M+H-H2O]+,[M+H-H4O2]+,[M+NH4]+,[M]+" \
    -e 'C[0-]N[0-]O[0-]H[0-]S[0-5]P[0-3]I[0-1]Cl[0-1]F[0-1]Br[0-1]' \
    --tree-timeout 300 \
    --compound-timeout 300 \
    write-summaries \
    --output $OUTFOLDER \

# After formula
# #--ppm-max-ms2 10  \ #fingerprint \ #structure \
