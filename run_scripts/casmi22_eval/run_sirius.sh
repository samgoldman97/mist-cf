
OUTFOLDER_RAW="results/sirius_predict_casmi22/sirius_1_raw/"
OUTFOLDER="results/sirius_predict_casmi22/sirius_1/"
mkdir -p $OUTFOLDER
mkdir -p $OUTFOLDER_RAW

password=$SIRIUS_PW
INPUT_FILE=data/casmi22/CASMI_processed.mgf
CORES=32


# Use structure module as well
$SIRIUS_PATH login -u samlg@mit.edu -p
$SIRIUS_PATH  \
    --cores $CORES \
    --output  $OUTFOLDER_RAW \
    --input $INPUT_FILE \
    formula  \
    -i "[M+H]+,[M+K]+,[M+Na]+,[M+H-H2O]+,[M+H-H4O2]+,[M+NH4]+,[M]+" \
    -e 'C[0-]N[0-]O[0-]H[0-]S[0-5]P[0-3]I[0-1]Cl[0-1]F[0-1]Br[0-1]' \
    --ppm-max 5.0 \
    --tree-timeout 300 \
    --compound-timeout 300 \
    fingerprint \
    structure \
    write-summaries \
    --output $OUTFOLDER \

# After formula
# #--ppm-max-ms2 10  \ #fingerprint \ #structure \
