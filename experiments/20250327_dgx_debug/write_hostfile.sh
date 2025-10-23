#!/bin/bash
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}  # Default to 8 if not set
mkdir -p "$(dirname "$0")/hostfiles"
# need to add the current slurm jobid to hostfile name so that we don't add to previous hostfile
hostfile="$(dirname "$0")/hostfiles/hosts_${SLURM_JOBID}"
# be extra sure we aren't appending to a previous hostfile
rm $hostfile &> /dev/null
# loop over the node names
for i in `scontrol show hostnames $SLURM_NODELIST`
do
    # host_ip=$(nslookup $i | awk '/^Address: / { print $2 }' | tail -n1)
    # add a line to the hostfile
    # echo $host_ip slots=$GPUS_PER_NODE >>$hostfile
    echo $i slots=$GPUS_PER_NODE >>$hostfile
done
