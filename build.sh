set -e
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

rm -f $SCRIPTPATH/*.gcda
rm -f $SCRIPTPATH/*.gcno
mkdir -p $SCRIPTPATH/out
rm -f $SCRIPTPATH/out/*

rm -rf $SCRIPTPATH/build

echo "Building standalone classifier"

cd $SCRIPTPATH

# build the Edge Impulse classifier
make -f Makefile.tflite

echo "Building standalone classifier OK"

# clear up
rm -f $SCRIPTPATH/*.gcda
rm -f $SCRIPTPATH/*.gcno
rm -rf $SCRIPTPATH/out
