# Setup Ray Cluster on SLURM
The idea consists in starting a scheduler in the main node (i.e. the one entered when logging in) and then send jobs which activate workers in the other nodes.

Start the scheduler:
```bash
$ ray start --head --port=6379 --num-cpus 1 --block
```
The file ```submit_worker.sub``` contains the job instructions to submit a worker in the node where it runs. 
To start e.g. 20 workers, type the following:
```bash
for i in {1..20}; do sbatch submit_worker.sub 192.33.123.23:6379 5241590000000000; done
```

NB: for some reason not known at the time of writing, a lot of them fail and/or don't last long; consider running the command many times until an appropriate number of active workers is reached. 
