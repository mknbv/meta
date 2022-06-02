
while true; do
    clear
    seq $1 $2 | xargs -L1 tsp -o | xargs tail -n1 \
        | awk -W interact 'NR % 2 == 0'
    sleep 2
done
