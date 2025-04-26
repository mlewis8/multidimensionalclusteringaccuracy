homeDir="./"
inputDir="./input"
scriptName="clusterGenerator.py"

# cols rows dim k
cols=5
rows=400
dim=2 
k=3
expRow=200


# Number of ranks per program
rankSizes=3

# Retrieve the length of the list
numPrograms=${#rankSizes[@]}

# Iterate over the number of programs

inputFile="$homeDir/input/PARTITION_rows_${rows}_cols_${cols}_procs_${rankSizes}.csv"
if [ ! -f ${inputFile}  ]; then
        echo "Creating th input file"
        execFile="${homeDir}/createInput.py"
        python ${execFile} ${inputDir} ${cols} ${rows} ${expRow} ${rankSizes}
fi


# Will run each program 10 times
for ((rowIter=0; rowIter<${rows}; rowIter += 200)); do
        echo "Starting a ${i} processor job !"
        endIndex=$((rowIter +200))
        mpirun -n ${rankSizes[$i]} python $scriptName $homeDir ${cols} ${rowIter} ${endIndex} ${dim} ${rows} ${expRow} &
        echo "***********************************"
        wait
done