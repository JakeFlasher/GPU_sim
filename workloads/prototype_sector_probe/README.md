# SM86 prototype sector probe

Compile with `/opt/cuda/bin/nvcc -O2 -arch=sm_86 sector_probe.cu -o
/tmp/gpu_sector_probe`. Profile one launch with:

```text
ncu --target-processes all --kernel-name sector_probe --launch-count 1 \
  --metrics gpu__time_duration,sm__cycles_elapsed,smsp__inst_executed_pipe_lsu,\
smsp__warps_eligible,l1tex__t_sectors_pipe_lsu_mem_global_op_ld,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st --csv --page raw --print-units base \
  /tmp/gpu_sector_probe 1 0 32 4096
```

The merged `sim-run --ncu-csv` observation path consumes the resulting raw
CSV. Selected SM86 observations are retained in
`observations/sector_probe_sm86.csv`.
