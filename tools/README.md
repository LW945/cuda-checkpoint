# Tools

## `checkpoint_timer.py`

Measure CUDA checkpoint timings for a running process without using CRIU.

```bash
python3 tools/checkpoint_timer.py --pid <pid>
```

The tool performs the explicit CUDA state transition sequence below and reports
per-step timings plus the total elapsed time:

```text
lock -> checkpoint -> restore -> unlock
```

Optional flags:

* `--timeout-ms <ms>` sets the lock timeout. The default is `30000`.
* `--device-map <old=new,...>` forwards a GPU remap to the restore step.
* `--json` prints machine-readable output.
