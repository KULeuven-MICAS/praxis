# Naxirzag

$\text{Naxirzag} = \text{SNAX} + \text{MLIR} + \text{ZigZag}$

## Get started:

[Install pixi](https://pixi.sh), then:
```shell
git clone --recursive git@github.com:KULeuven-MICAS/naxirzag.git
cd naxirzag
pixi shell
```
Check if it works!:

Either you can try out an example from `kernels`
```shell
cd kernels/streamer_matmul
snakemake -c 8 all
```
Or run the `tests`:
```shell
lit -vv tests/filecheck
```
