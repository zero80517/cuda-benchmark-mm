# cuda-benchmark-mm
Matrix multiplication benchmark on cuda

## How to run

Start **x64 Native Tools Command Prompt for VS 2019** (it's important that is used x64 version, cause you will get many errors with x86) and change a directory to the repository's directory.

Then run:

```cmd
nvcc -o main main.cu -lcublas
```

There will be created many files include main.exe. Run main.exe to start benchmark.

## Example of the output on Intel Core i7 and GTX 1050

```

N = 512:

        cublas calculation time:

                GPU calculation time 0.308224 msec, comparison correct

        cublas-Xt calculation time:

                GPU calculation time 0.380352 msec, comparison correct

        SMEM-2 kernel calculation time:

                GPU calculation time 1.537504 msec, comparison correct

N = 1024:

        cublas calculation time:

                GPU calculation time 1.605184 msec, comparison correct

        cublas-Xt calculation time:

                GPU calculation time 1.652736 msec, comparison correct

        SMEM-2 kernel calculation time:

                GPU calculation time 11.652160 msec, comparison correct

N = 2048:

        cublas calculation time:

                GPU calculation time 12.157952 msec, comparison correct

        cublas-Xt calculation time:

                GPU calculation time 12.222272 msec, comparison correct

        SMEM-2 kernel calculation time:

                GPU calculation time 92.653923 msec, comparison correct
```
