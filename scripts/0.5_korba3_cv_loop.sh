CMD=$1

for name in korba_reczna; do
    NAME=korba3-${name}
    echo $NAME
    for j in $(seq 0 $((CVN-1))); do
        echo $j
        sh -c "$CMD $NAME $j"
    done
    echo
done