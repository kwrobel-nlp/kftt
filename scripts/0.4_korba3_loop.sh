CMD=$1

for name in f19 korba_reczna nkjp1m; do
    NAME=korba3-${name}
    echo $NAME
    sh -c "$CMD $NAME"
    echo
done